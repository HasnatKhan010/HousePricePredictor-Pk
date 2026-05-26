





# Sir to run this GUI run these command in the same directory
#_____________________________________________________________________________
# pip install -r requirements.txt
# python app.py
#_____________________________________________________________________________





import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    import customtkinter as ctk
except ModuleNotFoundError:
    print("\nMissing package: customtkinter")
    print("Install requirements with:")
    print("python -m pip install customtkinter")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    missing = exc.name
    print(f"\nMissing package: {missing}")
    print("Install requirements with:")
    print("python -m pip install -r requirements.txt")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "zameen_islamabad.csv"


def parse_area_to_marla(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    if "kanal" in text:
        return amount * 20
    return amount


def yes_no_to_int(value):
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "available"}:
        return 1
    if text in {"no", "n", "false", "none", ""}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def format_price(price):
    if price >= 1e7:
        return f"{price / 1e7:.2f} Crore PKR"
    return f"{price / 1e5:.2f} Lakh PKR"


def clean_dataset(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["price", "area", "location", "bedrooms", "bathrooms"])
    df["area_marla"] = df["area"].apply(parse_area_to_marla)
    df = df.dropna(subset=["area_marla"])
    df = df[(df["price"] > 0) & (df["area_marla"] > 0)]

    for col in ["bedrooms", "bathrooms", "built_in_year", "store_rooms", "kitchens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_cols = [
        "parking_space",
        "servant_quarters",
        "drawing_room",
        "dining_room",
        "study_room",
        "prayer_room",
        "powder_room",
        "lounge_sitting_room",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(yes_no_to_int)

    current_year = 2026
    if "built_in_year" not in df.columns:
        df["built_in_year"] = np.nan
    df["property_age"] = current_year - df["built_in_year"]
    df.loc[(df["property_age"] < 0) | (df["property_age"] > 80), "property_age"] = np.nan

    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].clip(lower=1)
    df["area_log"] = np.log1p(df["area_marla"])
    df["area_squared"] = df["area_marla"] ** 2
    df["room_density"] = df["rooms_total"] / df["area_marla"].clip(lower=1)
    df["area_x_bedrooms"] = df["area_marla"] * df["bedrooms"]
    df["area_x_bathrooms"] = df["area_marla"] * df["bathrooms"]
    df["main_location"] = df["location"].astype(str).str.split(",").str[0].str.strip()
    df["property_type"] = df["property_type"].fillna("House").astype(str)
    df["location"] = df["location"].astype(str)
    return df


def train_from_csv(path):
    df = clean_dataset(path)

    numeric_features = [
        "area_marla",
        "area_log",
        "area_squared",
        "bedrooms",
        "bathrooms",
        "rooms_total",
        "bed_bath_ratio",
        "room_density",
        "area_x_bedrooms",
        "area_x_bathrooms",
        "built_in_year",
        "property_age",
        "parking_space",
        "servant_quarters",
        "store_rooms",
        "kitchens",
        "drawing_room",
        "dining_room",
        "study_room",
        "prayer_room",
        "powder_room",
        "lounge_sitting_room",
    ]
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = ["location", "main_location", "property_type"]

    X = df[numeric_features + categorical_features]
    y = np.log1p(df["price"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []
    trained = {}
    for name, regressor in candidates.items():
        model = Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])
        model.fit(X_train, y_train)
        log_pred = model.predict(X_test)
        pred = np.expm1(log_pred)
        actual = np.expm1(y_test)
        results.append(
            {
                "Model": name,
                "MAE": float(mean_absolute_error(actual, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                "R2": float(r2_score(actual, pred)),
            }
        )
        trained[name] = model

    metrics = pd.DataFrame(results).sort_values("RMSE", ascending=True)
    best_name = str(metrics.iloc[0]["Model"])
    defaults = df[numeric_features].median(numeric_only=True).to_dict()
    return {
        "model": trained[best_name],
        "best_name": best_name,
        "metrics": metrics,
        "df": df,
        "numeric_features": numeric_features,
        "defaults": defaults,
    }


def build_prediction_row(area, bedrooms, bathrooms, location, property_type, defaults):
    row = dict(defaults)
    area = float(area)
    bedrooms = float(bedrooms)
    bathrooms = float(bathrooms)
    rooms_total = bedrooms + bathrooms
    row.update(
        {
            "area_marla": area,
            "area_log": np.log1p(area),
            "area_squared": area**2,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "rooms_total": rooms_total,
            "bed_bath_ratio": bedrooms / max(bathrooms, 1),
            "room_density": rooms_total / max(area, 1),
            "area_x_bedrooms": area * bedrooms,
            "area_x_bathrooms": area * bathrooms,
            "location": location,
            "main_location": str(location).split(",")[0].strip(),
            "property_type": property_type,
        }
    )
    return pd.DataFrame([row])


class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Islamabad House Price Predictor")
        self.root.geometry("1240x760")
        self.root.minsize(1100, 700)

        self.artifacts = None
        self.location_counts = {}
        self._apply_theme()
        self._build_ui()
        self._set_form_state("disabled")
        threading.Thread(target=self.load_model, daemon=True).start()

    def _apply_theme(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.root.configure(bg="#0b1220")

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "Modern.Treeview",
            background="#0f172a",
            fieldbackground="#0f172a",
            foreground="#e2e8f0",
            rowheight=34,
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Modern.Treeview.Heading",
            background="#1e293b",
            foreground="#f8fafc",
            relief="flat",
            padding=(12, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Modern.Treeview",
            background=[("selected", "#4f46e5")],
            foreground=[("selected", "#ffffff")],
        )
        style.map(
            "Modern.Treeview.Heading",
            background=[("active", "#273449")],
        )

        style.configure(
            "TScrollbar",
            background="#0f172a",
            troughcolor="#0b1220",
            bordercolor="#0b1220",
            arrowcolor="#94a3b8",
            darkcolor="#0f172a",
            lightcolor="#0f172a",
        )

    def _build_ui(self):
        # Palette
        self.bg = "#0b1220"
        self.panel = "#111827"
        self.panel_soft = "#0f172a"
        self.border = "#1f2937"
        self.text = "#e5e7eb"
        self.muted = "#94a3b8"
        self.accent = "#6366f1"
        self.accent_hover = "#4f46e5"
        self.success = "#10b981"
        self.success_soft = "#064e3b"
        self.warn = "#f59e0b"

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.shell = ctk.CTkFrame(self.root, fg_color=self.bg, corner_radius=0)
        self.shell.grid(row=0, column=0, sticky="nsew")
        self.shell.grid_columnconfigure(0, weight=1)
        self.shell.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.shell, fg_color=self.panel, corner_radius=18, border_width=1, border_color=self.border)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 14))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Islamabad House Price Predictor",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=self.text,
        )
        title.grid(row=0, column=0, sticky="w", padx=22, pady=(18, 2))

        subtitle = ctk.CTkLabel(
            header,
            text="Modern CustomTkinter dashboard with live model training, validation, and polished prediction output.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=self.muted,
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=22, pady=(0, 16))

        header_right = ctk.CTkFrame(header, fg_color="transparent")
        header_right.grid(row=0, column=1, rowspan=2, sticky="e", padx=18, pady=18)

        self.mode_switch = ctk.CTkSwitch(
            header_right,
            text="Light mode",
            command=self._toggle_mode,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            progress_color=self.accent,
            button_color=self.accent,
            button_hover_color=self.accent_hover,
        )
        self.mode_switch.pack(anchor="e", pady=(0, 8))
        self.mode_switch.deselect()

        self.loading_chip = ctk.CTkLabel(
            header_right,
            text="Training model...",
            corner_radius=999,
            fg_color=self.warn,
            text_color="#111827",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            padx=14,
            pady=6,
        )
        self.loading_chip.pack(anchor="e")

        body = ctk.CTkFrame(self.shell, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 18))
        body.grid_columnconfigure(0, weight=1, uniform="cols")
        body.grid_columnconfigure(1, weight=1, uniform="cols")
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(
            body,
            fg_color=self.panel,
            corner_radius=18,
            border_width=1,
            border_color=self.border,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(
            body,
            fg_color="transparent",
        )
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        input_card = ctk.CTkFrame(
            left,
            fg_color=self.panel_soft,
            corner_radius=14,
            border_width=1,
            border_color=self.border,
        )
        input_card.pack(fill="x", padx=18, pady=(18, 12))
        input_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            input_card,
            text="Property Input Form",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=self.text,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        ctk.CTkLabel(
            input_card,
            text="Enter the property details below to estimate the house price.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=self.muted,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 10))

        self.area_var = tk.StringVar(value="10")
        self.bedrooms_var = tk.StringVar(value="4")
        self.bathrooms_var = tk.StringVar(value="3")
        self.location_var = tk.StringVar(value="")
        self.property_type_var = tk.StringVar(value="")
        self.result_var = tk.StringVar(value="Training model from CSV...")
        self.range_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Loading dataset and training models...")

        self._add_field(input_card, "Area (Marla)", self.area_var, 2)
        self._add_field(input_card, "Bedrooms", self.bedrooms_var, 3)
        self._add_field(input_card, "Bathrooms", self.bathrooms_var, 4)

        ctk.CTkLabel(
            input_card,
            text="Location",
            text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).grid(row=5, column=0, sticky="w", padx=18, pady=(10, 8))
        self.location_box = ctk.CTkComboBox(
            input_card,
            variable=self.location_var,
            values=[],
            height=38,
            corner_radius=10,
            fg_color="#162033",
            button_color=self.accent,
            button_hover_color=self.accent_hover,
            text_color=self.text,
            dropdown_fg_color=self.panel,
            dropdown_hover_color="#1f2a44",
            dropdown_text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.location_box.grid(row=5, column=1, sticky="ew", padx=(0, 18), pady=(10, 8))

        ctk.CTkLabel(
            input_card,
            text="Property Type",
            text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).grid(row=6, column=0, sticky="w", padx=18, pady=(10, 8))
        self.property_type_box = ctk.CTkComboBox(
            input_card,
            variable=self.property_type_var,
            values=[],
            height=38,
            corner_radius=10,
            fg_color="#162033",
            button_color=self.accent,
            button_hover_color=self.accent_hover,
            text_color=self.text,
            dropdown_fg_color=self.panel,
            dropdown_hover_color="#1f2a44",
            dropdown_text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.property_type_box.grid(row=6, column=1, sticky="ew", padx=(0, 18), pady=(10, 8))

        self.predict_button = ctk.CTkButton(
            input_card,
            text="Predict Price",
            height=44,
            corner_radius=12,
            fg_color=self.accent,
            hover_color=self.accent_hover,
            text_color="#ffffff",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            command=self.predict,
        )
        self.predict_button.grid(row=7, column=0, columnspan=2, sticky="ew", padx=18, pady=(18, 8))

        self.input_note = ctk.CTkLabel(
            input_card,
            text="Model summary appears on the right once training finishes.",
            text_color=self.muted,
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.input_note.grid(row=8, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 18))

        output_card = ctk.CTkFrame(
            left,
            fg_color=self.success_soft,
            corner_radius=14,
            border_width=1,
            border_color="#14532d",
        )
        output_card.pack(fill="x", padx=18, pady=(0, 18))
        output_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_card,
            text="Estimated House Price",
            text_color="#d1fae5",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 6))

        self.price_label = ctk.CTkLabel(
            output_card,
            textvariable=self.result_var,
            text_color="#ffffff",
            font=ctk.CTkFont(family="Segoe UI", size=30, weight="bold"),
        )
        self.price_label.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 4))

        self.range_label = ctk.CTkLabel(
            output_card,
            textvariable=self.range_var,
            text_color="#dcfce7",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.range_label.grid(row=2, column=0, sticky="w", padx=18, pady=(0, 2))

        self.detail_label = ctk.CTkLabel(
            output_card,
            textvariable=self.detail_var,
            text_color="#bbf7d0",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.detail_label.grid(row=3, column=0, sticky="w", padx=18, pady=(0, 16))

        metrics_card = ctk.CTkFrame(
            right,
            fg_color=self.panel,
            corner_radius=18,
            border_width=1,
            border_color=self.border,
        )
        metrics_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        metrics_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            metrics_card,
            text="Model Metrics Summary",
            text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 6))

        self.summary_label = ctk.CTkLabel(
            metrics_card,
            text="Waiting for training...",
            text_color=self.muted,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.summary_label.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 12))

        table_frame = ctk.CTkFrame(
            metrics_card,
            fg_color=self.panel_soft,
            corner_radius=12,
            border_width=1,
            border_color=self.border,
        )
        table_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        metrics_card.grid_rowconfigure(2, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = ("model", "mae", "rmse", "r2")
        self.metrics_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            style="Modern.Treeview",
            selectmode="browse",
            height=8,
        )
        self.metrics_tree.heading("model", text="Model")
        self.metrics_tree.heading("mae", text="MAE")
        self.metrics_tree.heading("rmse", text="RMSE")
        self.metrics_tree.heading("r2", text="R2")

        self.metrics_tree.column("model", width=170, anchor="w")
        self.metrics_tree.column("mae", width=115, anchor="e")
        self.metrics_tree.column("rmse", width=115, anchor="e")
        self.metrics_tree.column("r2", width=80, anchor="e")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=yscroll.set)
        self.metrics_tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        yscroll.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)

        info_card = ctk.CTkFrame(
            right,
            fg_color=self.panel,
            corner_radius=18,
            border_width=1,
            border_color=self.border,
        )
        info_card.grid(row=1, column=0, sticky="nsew")
        info_card.grid_columnconfigure(0, weight=1)
        info_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            info_card,
            text="Runtime Status",
            text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))

        self.status_label = ctk.CTkLabel(
            info_card,
            textvariable=self.status_var,
            text_color=self.muted,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            justify="left",
            wraplength=470,
        )
        self.status_label.grid(row=1, column=0, sticky="nw", padx=18, pady=(0, 10))

        self.help_label = ctk.CTkLabel(
            info_card,
            text=(
                "This interface keeps the original data cleaning, feature engineering, "
                "training, and prediction math intact while presenting a polished dashboard."
            ),
            text_color=self.muted,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            justify="left",
            wraplength=470,
        )
        self.help_label.grid(row=2, column=0, sticky="sw", padx=18, pady=(0, 18))

    def _add_field(self, parent, label, variable, row):
        ctk.CTkLabel(
            parent,
            text=label,
            text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).grid(row=row, column=0, sticky="w", padx=18, pady=(10, 8))
        entry = ctk.CTkEntry(
            parent,
            textvariable=variable,
            height=38,
            corner_radius=10,
            fg_color="#162033",
            border_color="#2b3648",
            text_color=self.text,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=(10, 8))

    def _toggle_mode(self):
        mode = "Light" if self.mode_switch.get() else "Dark"
        ctk.set_appearance_mode(mode)

    def _set_form_state(self, state):
        if state == "normal":
            self.predict_button.configure(state="normal")
            self.location_box.configure(state="normal")
            self.property_type_box.configure(state="normal")
        else:
            self.predict_button.configure(state="disabled")
            self.location_box.configure(state="disabled")
            self.property_type_box.configure(state="disabled")

    def load_model(self):
        try:
            if not CSV_PATH.exists():
                raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
            artifacts = train_from_csv(CSV_PATH)
            self.root.after(0, lambda: self.on_model_loaded(artifacts))
        except Exception as exc:
            self.root.after(0, lambda: self.on_model_error(exc))

    def on_model_loaded(self, artifacts):
        self.artifacts = artifacts
        df = artifacts["df"]
        self.location_counts = df["location"].value_counts().to_dict()

        locations = sorted(df["location"].dropna().astype(str).unique())
        property_types = sorted(df["property_type"].dropna().astype(str).unique())

        if locations:
            self.location_box.configure(values=locations)
            self.location_var.set(next((loc for loc in locations if "dha" in loc.lower()), locations[0]))
        else:
            self.location_box.configure(values=[""])
            self.location_var.set("")

        if property_types:
            self.property_type_box.configure(values=property_types)
            self.property_type_var.set(property_types[0])
        else:
            self.property_type_box.configure(values=["House"])
            self.property_type_var.set("House")

        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        for _, row in artifacts["metrics"].iterrows():
            self.metrics_tree.insert(
                "",
                "end",
                values=(
                    row["Model"],
                    f"{row['MAE']:,.0f}",
                    f"{row['RMSE']:,.0f}",
                    f"{row['R2']:.3f}",
                ),
            )

        self.summary_label.configure(
            text=f"Rows used: {len(df):,}    |    Best model: {artifacts['best_name']}"
        )
        self.result_var.set("Ready")
        self.range_var.set("Enter details and click Predict Price.")
        self.detail_var.set("")
        self.status_var.set("Model trained successfully from CSV.")
        self.loading_chip.configure(text="Model ready", fg_color=self.success, text_color="#ffffff")
        self._set_form_state("normal")

    def on_model_error(self, exc):
        self.status_var.set("Model training failed.")
        self.loading_chip.configure(text="Load failed", fg_color="#ef4444", text_color="#ffffff")
        messagebox.showerror("Error", str(exc))

    def predict(self):
        if self.artifacts is None:
            messagebox.showwarning("Please wait", "Model is still training.")
            return

        try:
            area = float(self.area_var.get())
            bedrooms = float(self.bedrooms_var.get())
            bathrooms = float(self.bathrooms_var.get())
            if area <= 0:
                raise ValueError("Area must be greater than 0.")
            if bedrooms + bathrooms <= 0:
                raise ValueError("Enter at least one bedroom or bathroom.")
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        location = self.location_var.get().strip()
        property_type = self.property_type_var.get().strip() or "House"

        row = build_prediction_row(
            area,
            bedrooms,
            bathrooms,
            location,
            property_type,
            self.artifacts["defaults"],
        )
        feature_order = self.artifacts["numeric_features"] + ["location", "main_location", "property_type"]
        log_prediction = self.artifacts["model"].predict(row[feature_order])[0]
        price = max(0, float(np.expm1(log_prediction)))

        frequency = int(self.location_counts.get(location, 0))
        margin_pct = 0.10 if frequency > 5 else 0.15 if frequency > 2 else 0.20

        self.result_var.set(format_price(price))
        self.range_var.set(
            f"Expected range: {format_price(price * (1 - margin_pct))} - {format_price(price * (1 + margin_pct))}"
        )
        self.detail_var.set(
            f"{property_type} in {location} | {area:g} Marla | {bedrooms:g} bed | {bathrooms:g} bath"
        )

        if frequency <= 2:
            self.status_var.set("Prediction completed. Note: this location has few examples in the CSV.")
        else:
            self.status_var.set("Prediction completed.")


def main():
    root = ctk.CTk()
    app = HousePriceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
