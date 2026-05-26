import time
import random
import re
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

TARGET_CITY  = "Islamabad"
START_URL    = "https://www.zameen.com/Homes/Islamabad-3-1.html"
MAX_PAGES    = 25
SAVE_FILE    = "zameen_islamabad.csv"
PAGE_PAUSE   = (3.5, 6.0)
DETAIL_PAUSE = (2.5, 4.5)

class HumanBrowser:
    def __init__(self):
        self.driver = self._launch()

    def _launch(self):
        options = uc.ChromeOptions()
        options.headless = False
        options.add_argument("--window-size=1366,768")
        
        try:
            return uc.Chrome(options=options)
        except Exception as e:
            match = re.search(r"Current browser version is (\d+)", str(e))
            if match:
                main_version = int(match.group(1))
                
                new_options = uc.ChromeOptions()
                new_options.headless = False
                new_options.add_argument("--window-size=1366,768")
                
                return uc.Chrome(options=new_options, version_main=main_version)
            raise e

    def open(self, url: str):
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, 12).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
        except Exception:
            pass
        self._human_pause(*PAGE_PAUSE)
        self._random_scroll()

    def open_detail(self, url: str):
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, 12).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
        except Exception:
            pass
        self._human_pause(*DETAIL_PAUSE)

    def html(self) -> str:
        return self.driver.page_source

    def _human_pause(self, low: float, high: float):
        time.sleep(random.uniform(low, high))

    def _random_scroll(self):
        depth = random.randint(400, 900)
        self.driver.execute_script(f"window.scrollBy(0, {depth})")
        time.sleep(random.uniform(0.6, 1.4))

    def quit(self):
        self.driver.quit()








        

class SearchPageParser:
    def __init__(self, html: str):
        self.soup = BeautifulSoup(html, "html.parser")

    def detail_urls(self) -> list[str]:
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
                    full = self._abs(href)
                    if full not in seen:
                        urls.append(full)
                        seen.add(full)
            return urls
        for card in cards:
            link = card.find("a", href=lambda h: h and "/Property/" in h)
            if link:
                full = self._abs(link["href"])
                if full not in seen:
                    urls.append(full)
                    seen.add(full)
        return urls

    def next_page_url(self, current_url: str) -> str | None:
        m = re.search(r"-(\d+)\.html$", current_url)
        if m:
            nxt = int(m.group(1)) + 1
            return re.sub(r"-\d+\.html$", f"-{nxt}.html", current_url)
        return None

    @staticmethod
    def _abs(href: str) -> str:
        href = href.split("?")[0]
        return href if href.startswith("http") else "https://www.zameen.com" + href

class DetailPageParser:
    AMENITY_MAP = {
        "parking": "parking_space",
        "servant": "servant_quarters",
        "store": "store_rooms",
        "kitchen": "kitchens",
        "drawing": "drawing_room",
        "dining": "dining_room",
        "dinning": "dining_room",
        "study": "study_room",
        "prayer": "prayer_room",
        "masjid": "prayer_room",
        "powder": "powder_room",
        "lounge": "lounge_sitting_room",
        "sitting": "lounge_sitting_room",
        "living": "lounge_sitting_room",
    }

    def __init__(self, html: str, url: str):
        self.soup = BeautifulSoup(html, "html.parser")
        self.url  = url

    def extract(self) -> dict:
        rec = {
            "url": self.url,
            "city": TARGET_CITY,
            "title": self._title(),
            "price": None,
            "location": self._location(),
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
        self._from_aria_labels(rec)
        self._from_text_patterns(rec)
        self._price_fallback(rec)
        return rec

    def _from_aria_labels(self, rec: dict):
        aria_map = {
            "Price": ("price", self._to_pkr),
            "Area": ("area", str),
            "Beds": ("bedrooms", self._to_int),
            "Baths": ("bathrooms", self._to_int),
            "Type": ("property_type", str),
        }
        for aria_val, (field, converter) in aria_map.items():
            el = self.soup.find(attrs={"aria-label": aria_val})
            if el:
                raw = el.get_text(" ", strip=True)
                try:
                    rec[field] = converter(raw)
                except Exception:
                    rec[field] = raw
        if not rec["location"]:
            loc_el = self.soup.find(attrs={"aria-label": "Location"})
            if loc_el:
                rec["location"] = loc_el.get_text(" ", strip=True)

    def _from_text_patterns(self, rec: dict):
        for tag in self.soup.find_all(["li", "span", "div"]):
            text = tag.get_text(" ", strip=True)
            if not text or len(text) > 120:
                continue
            lower = text.lower()
            if rec["built_in_year"] is None:
                yr = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", text)
                if yr:
                    rec["built_in_year"] = yr.group(1)
            if rec["bedrooms"] is None:
                b = re.search(r"(\d+)\s*bed", lower)
                if b:
                    rec["bedrooms"] = int(b.group(1))
            if rec["bathrooms"] is None:
                b = re.search(r"(\d+)\s*bath", lower)
                if b:
                    rec["bathrooms"] = int(b.group(1))
            for keyword, field in self.AMENITY_MAP.items():
                if keyword in lower and rec[field] is None:
                    n = self._to_int(text)
                    rec[field] = n if n else "Yes"

    def _price_fallback(self, rec: dict):
        if rec["price"] is not None:
            return
        price_re = re.compile(r"(pkr|rs\.?)\s*([\d,.]+)\s*(crore|lakh|million)?", re.I)
        for tag in self.soup.find_all(string=price_re):
            converted = self._to_pkr(str(tag))
            if converted:
                rec["price"] = converted
                break

    def _title(self) -> str | None:
        h1 = self.soup.find("h1")
        return h1.get_text(strip=True) if h1 else None

    def _location(self) -> str | None:
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
    def _to_pkr(raw: str) -> float | None:
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
        return base * {"crore": 1e7, "lakh": 1e5, "million": 1e6, "thousand": 1e3, "k": 1e3}.get(unit, 1)

    @staticmethod
    def _to_int(raw: str) -> int | None:
        m = re.search(r"\d+", str(raw))
        return int(m.group()) if m else None

class ZameenCollector:
    def __init__(self):
        self.browser   = HumanBrowser()
        self.records   = []
        self.seen      = set()

    def run(self, max_pages: int = MAX_PAGES):
        current_url = START_URL
        print(f"\nZameen Scraper | City: {TARGET_CITY} | Max pages: {max_pages}")
        for page_num in range(1, max_pages + 1):
            print(f"\nPage {page_num}/{max_pages} -> {current_url}")
            self.browser.open(current_url)
            page_parser  = SearchPageParser(self.browser.html())
            detail_links = [u for u in page_parser.detail_urls() if u not in self.seen]
            print(f"{len(detail_links)} new listings found")
            self._process_details(detail_links)
            print(f"Running total: {len(self.records)} records")
            next_url = page_parser.next_page_url(current_url)
            if not next_url:
                print("\nNo further pages — stopping.")
                break
            current_url = next_url
        self._save()
        self.browser.quit()
        print(f"\nComplete! {len(self.records)} listings -> {SAVE_FILE}\n")

    def _process_details(self, links: list[str]):
        for idx, url in enumerate(links, 1):
            self.seen.add(url)
            print(f"[{idx:>3}/{len(links)}]", end=" ")
            try:
                self.browser.open_detail(url)
                rec = DetailPageParser(self.browser.html(), url).extract()
                if rec["price"] and rec["area"]:
                    self.records.append(rec)
                    beds  = rec["bedrooms"]  or "?"
                    baths = rec["bathrooms"] or "?"
                    pkr_m = f"{rec['price']/1e6:.1f}M" if rec["price"] else "?"
                    print(f"PKR {pkr_m} | {rec['area']} | {beds}bed {baths}bath | {rec['location'] or 'location?'}")
                else:
                    print("Skipped — missing price or area")
            except Exception as err:
                print(f"{type(err).__name__}: {err}")
            if len(self.records) > 0 and len(self.records) % 50 == 0:
                self._save()
                print(f"\nCheckpoint saved ({len(self.records)} records)\n")

    def _save(self):
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
        pd.DataFrame(self.records)[cols].to_csv(SAVE_FILE, index=False, encoding="utf-8")

if __name__ == "__main__":
    ZameenCollector().run(max_pages=MAX_PAGES)