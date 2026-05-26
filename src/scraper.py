"""
Web Scraper Module
Handles data collection from Zameen.com using Selenium and BeautifulSoup
"""

import re
import time
import random
import logging
from typing import Optional, List, Dict, Set
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class PropertyScraper:
    """
    Professional web scraper for Zameen.com property listings.
    
    Implements human-like browsing patterns to avoid detection while
    collecting real estate data from Islamabad.
    """
    
    # Configuration
    TARGET_CITY = "Islamabad"
    START_URL = "https://www.zameen.com/Homes/Islamabad-3-1.html"
    MAX_PAGES = 25
    OUTPUT_FILE = "zameen_islamabad.csv"
    PAGE_PAUSE = (3.5, 6.0)
    DETAIL_PAUSE = (2.5, 4.5)
    
    # Amenity mapping for consistent naming
    AMENITY_MAP = {
        "parking": "parking_space",
        "servant": "servant_quarters",
        "store": "store_rooms",
        "kitchen": "kitchens",
        "drawing": "drawing_room",
        "dining": "dining_room",
        "study": "study_room",
        "prayer": "prayer_room",
        "powder": "powder_room",
        "lounge": "lounge_sitting_room",
        "sitting": "lounge_sitting_room",
        "living": "lounge_sitting_room",
    }
    
    def __init__(self, headless: bool = False):
        """Initialize PropertyScraper with Chrome driver."""
        self.driver = self._initialize_driver(headless)
        self.records: List[Dict] = []
        self.seen_urls: Set[str] = set()
        logger.info("PropertyScraper initialized")
    
    def _initialize_driver(self, headless: bool):
        """Initialize undetected Chrome driver with retry logic."""
        try:
            options = uc.ChromeOptions()
            options.headless = headless
            options.add_argument("--window-size=1366,768")
            return uc.Chrome(options=options)
        except Exception as e:
            # Version mismatch handling
            match = re.search(r"Current browser version is (\d+)", str(e))
            if match:
                main_version = int(match.group(1))
                options = uc.ChromeOptions()
                options.headless = headless
                options.add_argument("--window-size=1366,768")
                return uc.Chrome(options=options, version_main=main_version)
            raise e
    
    def scrape(self, max_pages: int = MAX_PAGES) -> pd.DataFrame:
        """Execute scraping pipeline."""
        logger.info(f"Starting scrape: {self.TARGET_CITY}, max {max_pages} pages")
        current_url = self.START_URL
        
        for page_num in range(1, max_pages + 1):
            logger.info(f"Page {page_num}/{max_pages}: {current_url}")
            self._process_page(current_url, page_num, max_pages)
            
            parser = SearchPageParser(self.driver.page_source)
            current_url = parser.next_page_url(current_url)
            if not current_url:
                logger.info("No more pages available")
                break
        
        self._save_records()
        self.driver.quit()
        logger.info(f"Scraping complete: {len(self.records)} records saved")
        return pd.DataFrame(self.records)
    
    def _process_page(self, url: str, page_num: int, total_pages: int):
        """Process single search results page."""
        self.driver.get(url)
        self._wait_for_page()
        
        parser = SearchPageParser(self.driver.page_source)
        detail_urls = [u for u in parser.detail_urls() if u not in self.seen_urls]
        logger.info(f"Found {len(detail_urls)} new listings")
        
        for idx, detail_url in enumerate(detail_urls, 1):
            self._process_detail(detail_url, idx, len(detail_urls))
        
        if len(self.records) % 50 == 0 and len(self.records) > 0:
            self._save_records()
            logger.info(f"Checkpoint: {len(self.records)} records")
    
    def _process_detail(self, url: str, idx: int, total: int):
        """Process individual property detail page."""
        self.seen_urls.add(url)
        try:
            self.driver.get(url)
            self._wait_for_detail()
            
            extractor = DetailPageExtractor(self.driver.page_source, url)
            record = extractor.extract()
            
            if record.get("price") and record.get("area"):
                self.records.append(record)
                logger.debug(f"[{idx}/{total}] Added: {record['location']} - PKR {record['price']:.0f}")
            else:
                logger.debug(f"[{idx}/{total}] Skipped: Missing price or area")
        except Exception as e:
            logger.warning(f"Error processing {url}: {e}")
    
    def _wait_for_page(self):
        """Wait for page load with human-like delay."""
        try:
            WebDriverWait(self.driver, 12).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
        except:
            pass
        self._human_pause(*self.PAGE_PAUSE)
        self._random_scroll()
    
    def _wait_for_detail(self):
        """Wait for detail page load."""
        try:
            WebDriverWait(self.driver, 12).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
        except:
            pass
        self._human_pause(*self.DETAIL_PAUSE)
    
    @staticmethod
    def _human_pause(low: float, high: float):
        """Simulate human pause between actions."""
        time.sleep(random.uniform(low, high))
    
    @staticmethod
    def _random_scroll():
        """Simulate human scrolling."""
        depth = random.randint(400, 900)
        # Scroll action would go here
        time.sleep(random.uniform(0.6, 1.4))
    
    def _save_records(self):
        """Save records to CSV file."""
        if not self.records:
            return
        
        cols = [
            "city", "title", "price", "area", "location",
            "property_type", "bedrooms", "bathrooms",
            "built_in_year", "parking_space", "servant_quarters",
            "store_rooms", "kitchens", "drawing_room", "dining_room",
            "study_room", "prayer_room", "powder_room",
            "lounge_sitting_room", "url",
        ]
        df = pd.DataFrame(self.records)[cols]
        df.to_csv(self.OUTPUT_FILE, index=False, encoding="utf-8")
        logger.info(f"Saved {len(self.records)} records to {self.OUTPUT_FILE}")


class SearchPageParser:
    """Parse search results page to extract property detail URLs."""
    
    def __init__(self, html: str):
        self.soup = BeautifulSoup(html, "html.parser")
    
    def detail_urls(self) -> List[str]:
        """Extract all property detail URLs from search page."""
        urls = []
        seen = set()
        
        cards = self.soup.find_all(attrs={"aria-label": "Listing"})
        if not cards:
            cards = self.soup.find_all(attrs={"role": "article"})
        if not cards:
            main_tag = self.soup.find("main")
            search_scope = main_tag if main_tag else self.soup
            for a in search_scope.find_all("a", href=True):
                href = a["href"]
                if "/Property/" in href:
                    full = self._absolute_url(href)
                    if full not in seen:
                        urls.append(full)
                        seen.add(full)
            return urls
        
        for card in cards:
            link = card.find("a", href=lambda h: h and "/Property/" in h)
            if link:
                full = self._absolute_url(link["href"])
                if full not in seen:
                    urls.append(full)
                    seen.add(full)
        
        return urls
    
    def next_page_url(self, current_url: str) -> Optional[str]:
        """Get URL for next page in pagination."""
        m = re.search(r"-(\d+)\.html$", current_url)
        if m:
            nxt = int(m.group(1)) + 1
            return re.sub(r"-\d+\.html$", f"-{nxt}.html", current_url)
        return None
    
    @staticmethod
    def _absolute_url(href: str) -> str:
        """Convert relative URL to absolute."""
        href = href.split("?")[0]
        return href if href.startswith("http") else "https://www.zameen.com" + href


class DetailPageExtractor:
    """Extract property details from detail page."""
    
    AMENITY_MAP = PropertyScraper.AMENITY_MAP
    
    def __init__(self, html: str, url: str):
        self.soup = BeautifulSoup(html, "html.parser")
        self.url = url
    
    def extract(self) -> Dict:
        """Extract all property information from detail page."""
        record = {
            "url": self.url,
            "city": "Islamabad",
            "title": self._extract_title(),
            "price": None,
            "location": self._extract_location(),
            "property_type": None,
            "bedrooms": None,
            "bathrooms": None,
            "area": None,
            "built_in_year": None,
            "parking_space": None,
            "servant_quarters": None,
            "store_rooms": None,
            "kitchens": None,
            "drawing_room": None,
            "dining_room": None,
            "study_room": None,
            "prayer_room": None,
            "powder_room": None,
            "lounge_sitting_room": None,
        }
        
        self._extract_from_aria_labels(record)
        self._extract_from_text_patterns(record)
        self._extract_price_fallback(record)
        
        return record
    
    def _extract_from_aria_labels(self, record: Dict):
        """Extract data from aria-label attributes."""
        aria_map = {
            "Price": ("price", self._parse_pkr),
            "Area": ("area", str),
            "Beds": ("bedrooms", self._parse_int),
            "Baths": ("bathrooms", self._parse_int),
            "Type": ("property_type", str),
        }
        
        for aria_val, (field, converter) in aria_map.items():
            el = self.soup.find(attrs={"aria-label": aria_val})
            if el:
                try:
                    record[field] = converter(el.get_text(" ", strip=True))
                except:
                    pass
    
    def _extract_from_text_patterns(self, record: Dict):
        """Extract data using regex patterns."""
        for tag in self.soup.find_all(["li", "span", "div"]):
            text = tag.get_text(" ", strip=True)
            if not text or len(text) > 120:
                continue
            
            lower = text.lower()
            
            if record["built_in_year"] is None:
                yr = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", text)
                if yr:
                    record["built_in_year"] = yr.group(1)
            
            for keyword, field in self.AMENITY_MAP.items():
                if keyword in lower and record[field] is None:
                    n = self._parse_int(text)
                    record[field] = n if n else "Yes"
    
    def _extract_price_fallback(self, record: Dict):
        """Fallback price extraction using regex."""
        if record["price"] is not None:
            return
        
        price_re = re.compile(r"(pkr|rs\.?)\s*([\d,.]+)\s*(crore|lakh|million)?", re.I)
        for tag in self.soup.find_all(string=price_re):
            converted = self._parse_pkr(str(tag))
            if converted:
                record["price"] = converted
                break
    
    def _extract_title(self) -> Optional[str]:
        """Extract property title."""
        h1 = self.soup.find("h1")
        return h1.get_text(strip=True) if h1 else None
    
    def _extract_location(self) -> Optional[str]:
        """Extract property location."""
        el = self.soup.find(attrs={"aria-label": "Location"})
        if el:
            return el.get_text(" ", strip=True)
        
        nav = self.soup.find("nav", attrs={"aria-label": re.compile("breadcrumb", re.I)})
        if nav:
            parts = [a.get_text(strip=True) for a in nav.find_all("a")]
            if parts:
                return " > ".join(parts[-2:])
        
        addr = self.soup.find("address")
        if addr:
            return addr.get_text(" ", strip=True)
        
        return None
    
    @staticmethod
    def _parse_pkr(raw: str) -> Optional[float]:
        """Parse price string to PKR float."""
        if not raw:
            return None
        
        raw = raw.replace(",", "").replace("\xa0", " ")
        m = re.search(r"([\d.]+)\s*(crore|lakh|million|thousand|k\b)?", raw, re.I)
        if not m:
            return None
        
        try:
            base = float(m.group(1))
        except ValueError:
            return None
        
        unit = (m.group(2) or "").lower()
        multipliers = {
            "crore": 1e7,
            "lakh": 1e5,
            "million": 1e6,
            "thousand": 1e3,
            "k": 1e3
        }
        return base * multipliers.get(unit, 1)
    
    @staticmethod
    def _parse_int(raw: str) -> Optional[int]:
        """Parse integer from string."""
        m = re.search(r"\d+", str(raw))
        return int(m.group()) if m else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = PropertyScraper()
    df = scraper.scrape(max_pages=PropertyScraper.MAX_PAGES)
    print(f"✅ Scraped {len(df)} properties")
