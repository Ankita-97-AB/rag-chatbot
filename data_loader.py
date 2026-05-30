import os
import pandas as pd
import fitz  # PyMuPDF


# ── PDF ───────────────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> list[dict]:
    chunks = []
    doc = fitz.open(file_path)
    fname = os.path.basename(file_path)
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            chunks.append({
                "source": fname,
                "type": "pdf",
                "page": f"Page {page_num}",
                "content": text,
            })
    doc.close()
    return chunks


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _rows_to_text(df_slice: pd.DataFrame) -> str:
    lines = []
    for _, row in df_slice.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items()
                 if pd.notna(val) and str(val).strip()]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def load_csv_cardata(file_path: str) -> list[dict]:
    """cardata.csv – Indian used car sales (Year/Selling_Price/Kms_Driven…)"""
    df = pd.read_csv(file_path)
    fname = os.path.basename(file_path)
    chunks, chunk_size = [], 15
    for i in range(0, len(df), chunk_size):
        subset = df.iloc[i: i + chunk_size]
        content = (
            "Used car records (India) – fields: Car_Name, Year, Selling_Price, "
            "Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner\n"
            + _rows_to_text(subset)
        )
        chunks.append({"source": fname, "type": "csv",
                        "page": f"rows {i+1}-{min(i+chunk_size, len(df))}",
                        "content": content})
    return chunks


def load_csv_prediction(file_path: str) -> list[dict]:
    """car_price_prediction.csv – global used car data (Brand/Engine/Price…)"""
    df = pd.read_csv(file_path)
    fname = os.path.basename(file_path)
    chunks, chunk_size = [], 20
    for i in range(0, len(df), chunk_size):
        subset = df.iloc[i: i + chunk_size]
        content = (
            "Car price prediction dataset (global) – fields: Car ID, Brand, Year, "
            "Engine Size, Fuel Type, Transmission, Mileage, Condition, Price, Model\n"
            + _rows_to_text(subset)
        )
        chunks.append({"source": fname, "type": "csv",
                        "page": f"rows {i+1}-{min(i+chunk_size, len(df))}",
                        "content": content})
    return chunks


def load_csv_cardekho(file_path: str) -> list[dict]:
    """cardekho.csv – CarDekho India listings (name/mileage/engine/seats…)"""
    df = pd.read_csv(file_path)
    fname = os.path.basename(file_path)
    chunks, chunk_size = [], 20
    for i in range(0, len(df), chunk_size):
        subset = df.iloc[i: i + chunk_size]
        content = (
            "CarDekho used car listings (India) – fields: name, year, selling_price, "
            "km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats\n"
            + _rows_to_text(subset)
        )
        chunks.append({"source": fname, "type": "csv",
                        "page": f"rows {i+1}-{min(i+chunk_size, len(df))}",
                        "content": content})
    return chunks


def load_csv_used_cars(file_path: str) -> list[dict]:
    """used_cars.csv – US used car listings (brand/model/milage/accident…)"""
    df = pd.read_csv(file_path)
    fname = os.path.basename(file_path)
    chunks, chunk_size = [], 20
    for i in range(0, len(df), chunk_size):
        subset = df.iloc[i: i + chunk_size]
        content = (
            "Used car listings (US market) – fields: brand, model, model_year, milage, "
            "fuel_type, engine, transmission, ext_col, int_col, accident, clean_title, price\n"
            + _rows_to_text(subset)
        )
        chunks.append({"source": fname, "type": "csv",
                        "page": f"rows {i+1}-{min(i+chunk_size, len(df))}",
                        "content": content})
    return chunks


# ── CSV pattern matcher ───────────────────────────────────────────────────────
# Maps keyword patterns (matched against filename) to loader functions.
# This works even if files have prefixes like "1780151843699_cardata.csv".

CSV_PATTERNS = [
    ("car_price_prediction", load_csv_prediction),   # check this BEFORE cardata
    ("cardata",              load_csv_cardata),
    ("cardekho",             load_csv_cardekho),
    ("used_cars",            load_csv_used_cars),
]


def _match_csv_loader(fname: str):
    """Return the right loader for a CSV filename, or None if unknown."""
    lower = fname.lower()
    for keyword, loader in CSV_PATTERNS:
        if keyword in lower:
            return loader
    return None


# ── Dynamic summary ───────────────────────────────────────────────────────────

def make_summary_chunks(data_dir: str) -> list[dict]:
    """Build summary chunks describing all files actually present in data/."""
    if not os.path.exists(data_dir):
        return []

    found = sorted(os.listdir(data_dir))
    csv_files  = [f for f in found if f.lower().endswith(".csv")]
    pdf_files  = [f for f in found if f.lower().endswith(".pdf")]

    summaries = []

    # Per-file summaries
    for fname in csv_files:
        loader = _match_csv_loader(fname)
        if loader:
            desc = loader.__doc__ or fname
            summaries.append({
                "source": fname,
                "type": "summary",
                "page": "Dataset Overview",
                "content": f"DATASET SUMMARY — {fname}\n{desc}",
            })

    for fname in pdf_files:
        lower = fname.lower()
        if "swift" in lower:
            desc = ("Maruti Suzuki New Swift official Chennai price list (2025). "
                    "Variants: LXI, VXI, VXI(O), ZXI, ZXI+ in 5MT and AGS, plus CNG. "
                    "Shows Ex-showroom, insurance, road tax, on-road prices in INR.")
        elif "hyundai" in lower or "model" in lower or "pricelist" in lower or "march" in lower:
            desc = ("KUN Hyundai Chennai price list (March 2024). Covers: "
                    "Grand i10 Nios, Aura, Exter, i20, i20 N Line, Venue, Creta, "
                    "Alcazar, Verna, Tucson, Kona Electric. On-road prices in INR.")
        else:
            desc = f"PDF document: {fname}"
        summaries.append({
            "source": fname,
            "type": "summary",
            "page": "Dataset Overview",
            "content": f"DATASET SUMMARY — {fname}\n{desc}",
        })

    # Combined overview
    all_files = "\n".join(f"  • {f}" for f in csv_files + pdf_files)
    summaries.append({
        "source": "ALL DATASETS",
        "type": "summary",
        "page": "Combined Overview",
        "content": (
            f"This car chatbot has access to {len(csv_files + pdf_files)} data files:\n"
            f"{all_files}\n"
            "You can ask about: used car prices (India/US), new car prices (Maruti/Hyundai), "
            "specific models, fuel types, transmissions, engine specs, and market comparisons."
        ),
    })

    return summaries


# ── Main loader ───────────────────────────────────────────────────────────────

def load_all_files(data_dir: str) -> list[dict]:
    """Load all supported files from data_dir. CSVs matched by name pattern, PDFs loaded all."""
    all_chunks = make_summary_chunks(data_dir)

    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        lower = fname.lower()
        try:
            if lower.endswith(".csv"):
                loader = _match_csv_loader(fname)
                if loader:
                    chunks = loader(fpath)
                    all_chunks.extend(chunks)
                    print(f" {fname}: {len(chunks)} chunks")
                else:
                    print(f" {fname}: no matching CSV loader (skipped)")

            elif lower.endswith(".pdf"):
                chunks = load_pdf(fpath)
                all_chunks.extend(chunks)
                print(f" {fname}: {len(chunks)} chunks (PDF)")

        except Exception as e:
            print(f" {fname}: {e}")

    print(f"\n  Total: {len(all_chunks)} chunks")
    return all_chunks