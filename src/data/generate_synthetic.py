"""
ArthaMart — Synthetic data generator.
Produces a realistic Indian-retail dataset:
  - ~65 products (MRP-locked packaged goods + BULK loose commodities)
  - 180 days of daily sales history with seasonality, festival uplifts, stockouts
  - Wholesale (mandi) price history with dips/spikes for bulk goods
  - ~3,000 transactions with correlated co-purchase pairs (for FP-Growth)
  - Customers for the personalisation engine
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .. import config
from . import database

fake = Faker("en_IN")
Faker.seed(config.get("project.seed", 42))
random.seed(config.get("project.seed", 42))
np.random.seed(config.get("project.seed", 42))

# ── Product catalogue ───────────────────────────────────────
MRP_PRODUCTS = [
    ("Biscuit Parle-G 800g",       "Snacks",    95,  68, 180, "pcs"),
    ("Britannia Good Day 600g",    "Snacks",    80,  56, 150, "pcs"),
    ("Sunfeast Yippee Noodles",    "Snacks",    70,  48, 210, "pcs"),
    ("Lays Classic Salted 52g",    "Snacks",    20,  13, 120, "pcs"),
    ("Kurkure Masala Munch 90g",   "Snacks",    25,  16, 120, "pcs"),
    ("Dairy Milk Silk 150g",       "Confectionery", 180, 125, 240, "pcs"),
    ("KitKat 37.3g",               "Confectionery", 35,  24, 270, "pcs"),
    ("Haldiram Bhujia 200g",       "Namkeen",   65,  44, 150, "pcs"),
    ("Haldiram Aloo Bhujia 400g",  "Namkeen",  120,  82, 150, "pcs"),
    ("Bikaji Bhujia 1kg",          "Namkeen",  280, 195, 180, "pcs"),
    ("Surf Excel Detergent 1kg",   "Household", 199, 142, 540, "pcs"),
    ("Tide Plus 1kg",              "Household", 155, 108, 540, "pcs"),
    ("Vim Dishwash Bar 500g",      "Household",  45,  30, 365, "pcs"),
    ("Colgate MaxFresh 300g",      "Personal Care", 165, 112, 420, "pcs"),
    ("Dove Cream Beauty 125g",     "Personal Care", 70,  48, 365, "pcs"),
    ("Lifebuoy Soap 125g",         "Personal Care", 38,  25, 365, "pcs"),
    ("Clinic Plus Shampoo 340ml",  "Personal Care", 210, 145, 365, "pcs"),
    ("Head & Shoulders 340ml",     "Personal Care", 330, 230, 365, "pcs"),
    ("Brooke Bond Red Label 500g", "Beverages", 270, 188, 365, "pcs"),
    ("Tata Tea Premium 500g",      "Beverages", 320, 225, 365, "pcs"),
    ("Nescafe Classic 100g",       "Beverages", 345, 245, 540, "pcs"),
    ("Complan 500g",               "Beverages", 310, 218, 365, "pcs"),
    ("Maggi Masala 70g (pack-12)", "Grocery",    168, 115, 270, "pcs"),
    ("Aashirvaad Atta 5kg",        "Grocery",    335, 252, 120, "pcs"),
    ("Fortune Sunflower Oil 1L",   "Grocery",    180, 142, 365, "pcs"),
    ("Saffola Gold Oil 1L",        "Grocery",    195, 152, 365, "pcs"),
    ("Amul Butter 500g",           "Dairy",      270, 205, 90,  "pcs"),
    ("Amul Cheese Slices 200g",    "Dairy",      135, 102, 120, "pcs"),
    ("Mother Dairy Curd 400g",     "Dairy",       50,  35, 14,  "pcs"),
    ("Real Mango Juice 1L",        "Beverages",  120,  82, 180, "pcs"),
    ("Coca-Cola 750ml",            "Beverages",   40,  26, 120, "pcs"),
    ("Pepsi 750ml",                "Beverages",   40,  26, 120, "pcs"),
    ("Bisleri Water 2L",           "Beverages",   30,  18, 365, "pcs"),
    ("Red Bull 250ml",             "Beverages",  125,  88, 270, "pcs"),
    ("Pedigree Dog Food 1kg",      "Pet Care",   375, 268, 540, "pcs"),
]

BULK_PRODUCTS = [
    ("Rice Basmati (loose)",    "Grain",  85, "kg"),
    ("Rice Sona Masoori (loose)","Grain", 55, "kg"),
    ("Wheat Atta (loose)",      "Grain",  42, "kg"),
    ("Toor Dal (loose)",        "Pulses", 135, "kg"),
    ("Moong Dal (loose)",       "Pulses", 110, "kg"),
    ("Chana Dal (loose)",       "Pulses", 75, "kg"),
    ("Masoor Dal (loose)",      "Pulses", 95, "kg"),
    ("Sugar (loose)",           "Grocery",48, "kg"),
    ("Salt (loose)",            "Grocery",18, "kg"),
    ("Tea Powder (loose)",      "Beverages", 280, "kg"),
    ("Coffee Powder (loose)",   "Beverages", 550, "kg"),
    ("Mustard Oil (loose)",     "Oil",    145, "L"),
    ("Groundnut Oil (loose)",   "Oil",    195, "L"),
    ("Coconut Oil (loose)",     "Oil",    220, "L"),
    ("Red Chilli Powder (loose)","Spice", 320, "kg"),
    ("Turmeric Powder (loose)", "Spice",  240, "kg"),
    ("Coriander Powder (loose)","Spice",  220, "kg"),
    ("Garam Masala (loose)",    "Spice",  480, "kg"),
    ("Jeera (loose)",           "Spice",  420, "kg"),
    ("Cashew (loose)",          "Dry Fruit", 750, "kg"),
    ("Almonds (loose)",         "Dry Fruit", 880, "kg"),
    ("Raisins (loose)",         "Dry Fruit", 320, "kg"),
    ("Onion (loose)",           "Vegetable", 30, "kg"),
    ("Potato (loose)",          "Vegetable", 25, "kg"),
    ("Tomato (loose)",          "Vegetable", 40, "kg"),
    ("Banana (loose)",          "Fruit",   50, "kg"),
    ("Apple (loose)",           "Fruit",  120, "kg"),
    ("Jaggery (loose)",         "Grocery", 65, "kg"),
    ("Poha (loose)",            "Grocery", 55, "kg"),
    ("Rava (loose)",            "Grocery", 48, "kg"),
]

# Co-purchase correlation groups (for basket signal)
BASKET_GROUPS = [
    ["Tea Powder (loose)", "Biscuit Parle-G 800g"],
    ["Rice Basmati (loose)", "Toor Dal (loose)"],
    ["Wheat Atta (loose)", "Sugar (loose)"],
    ["Coca-Cola 750ml", "Lays Classic Salted 52g"],
    ["Nescafe Classic 100g", "Dairy Milk Silk 150g"],
    ["Aashirvaad Atta 5kg", "Fortune Sunflower Oil 1L"],
    ["Maggi Masala 70g (pack-12)", "Coca-Cola 750ml"],
    ["Red Chilli Powder (loose)", "Turmeric Powder (loose)"],
    ["Onion (loose)", "Potato (loose)", "Tomato (loose)"],
    ["Amul Butter 500g", "Aashirvaad Atta 5kg"],
    ["Brooke Bond Red Label 500g", "Biscuit Parle-G 800g"],
    ["Toor Dal (loose)", "Ghee (loose)" if False else "Rice Basmati (loose)"],
    ["Colgate MaxFresh 300g", "Dove Cream Beauty 125g"],
    ["Surf Excel Detergent 1kg", "Vim Dishwash Bar 500g"],
    ["Pepsi 750ml", "Kurkure Masala Munch 90g"],
]

# Festival months (sales uplift) — Oct=Diwali, Jan=Sankranti, Aug=Independence/Raksha
FESTIVAL_MONTHS = {8: 1.25, 10: 1.45, 11: 1.15, 1: 1.20}


def generate_products():
    """Build list of product dicts."""
    products = []
    for i, (name, cat, mrp, cost, shelf, unit) in enumerate(MRP_PRODUCTS, 1):
        products.append({
            "sku": f"SKU-M{i:03d}",
            "name": name, "category": cat, "regime": "MRP",
            "mrp": mrp, "wholesale_cost": cost,
            "base_margin_pct": round((mrp - cost) / mrp * 100, 1),
            "shelf_life_days": shelf, "unit": unit,
        })
    for i, (name, cat, cost, unit) in enumerate(BULK_PRODUCTS, 1):
        mrp = round(cost * 1.22, 0)  # 22% base margin
        products.append({
            "sku": f"SKU-B{i:03d}",
            "name": name, "category": cat, "regime": "BULK",
            "mrp": None, "wholesale_cost": cost,
            "base_margin_pct": 22.0,
            "shelf_life_days": None, "unit": unit,
        })
    return products


def generate_daily_sales(products, start_date, n_days):
    """Generate daily sales with weekly seasonality, festivals, occasional stockouts."""
    rows = []
    # Assign product_id = index + 1 (matches SQLite AUTOINCREMENT insertion order)
    for idx, prod in enumerate(products, 1):
        prod["product_id"] = idx

    for d in range(n_days):
        date = start_date + timedelta(days=d)
        dow = date.weekday()
        weekend = 1.35 if dow >= 5 else 1.0
        festival = FESTIVAL_MONTHS.get(date.month, 1.0)
        trend = 1.0 + (d / n_days) * 0.10  # 10% growth over period

        for prod in products:
            # Base velocity depends on category
            base = _base_velocity(prod)
            noise = np.random.normal(1.0, 0.25)
            lam = base * weekend * festival * trend * max(noise, 0.1)
            qty = int(np.random.poisson(lam))

            # Occasional stockouts (~5% of days for some products)
            stockout = 1 if (qty > 0 and random.random() < 0.04) else 0
            sold = 0 if stockout else qty

            price = prod["mrp"] if prod["regime"] == "MRP" else prod["wholesale_cost"] * 1.22
            revenue = round(sold * price, 2)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": prod["product_id"],
                "qty_sold": sold, "revenue": revenue, "stockout": stockout,
            })
    return rows


def _base_velocity(prod) -> float:
    """Average daily sales units for a product category."""
    v = {
        "Snacks": 8, "Confectionery": 5, "Namkeen": 6, "Household": 4,
        "Personal Care": 5, "Beverages": 7, "Grocery": 6, "Dairy": 9,
        "Pet Care": 2, "Grain": 12, "Pulses": 6, "Oil": 5,
        "Spice": 3, "Dry Fruit": 2, "Vegetable": 15, "Fruit": 10,
    }
    return v.get(prod["category"], 5)


def generate_wholesale_prices(products, start_date, n_days):
    """Generate mandi/wholesale cost history — fluctuates for BULK, stable for MRP."""
    rows = []
    for prod in products:
        base_cost = prod["wholesale_cost"]
        for d in range(n_days):
            date = start_date + timedelta(days=d)
            if prod["regime"] == "BULK":
                # Mean-reverting random walk with occasional dips/spikes
                shock = np.random.normal(1.0, 0.04)
                dip = 0.88 if random.random() < 0.03 else 1.0
                spike = 1.18 if random.random() < 0.02 else 1.0
                cost = round(base_cost * shock * dip * spike, 2)
            else:
                cost = base_cost  # MRP goods have stable wholesale cost
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": prod["product_id"],
                "cost": cost,
            })
    return rows


def generate_transactions(products, daily_sales_df, n_transactions):
    """Generate ~n_transactions baskets with correlated co-purchase pairs."""
    name_to_id = {p["name"]: p["product_id"] for p in products}
    # Build correlation map
    corr_pairs = []
    for group in BASKET_GROUPS:
        ids = [name_to_id[n] for n in group if n in name_to_id]
        if len(ids) >= 2:
            corr_pairs.append(ids)

    rows = []
    customers = [i for i in range(1, 201)]  # 200 customers
    start = datetime.strptime(config.get("data.start_date"), "%Y-%m-%d")

    for t in range(n_transactions):
        date = start + timedelta(days=random.randint(0, config.get("data.n_days") - 1))
        customer_id = random.choice(customers)

        # Pick a correlated group sometimes
        basket = set()
        if random.random() < 0.45 and corr_pairs:
            group = random.choice(corr_pairs)
            for pid in group:
                if random.random() < 0.75:
                    basket.add(pid)
        # Add 1-5 random additional products
        for _ in range(random.randint(1, 5)):
            basket.add(random.choice(products)["product_id"])

        for pid in basket:
            prod = next(p for p in products if p["product_id"] == pid)
            qty = random.randint(1, 4)
            price = prod["mrp"] or prod["wholesale_cost"] * 1.22
            rows.append({
                "transaction_id": t + 1,
                "date": date.strftime("%Y-%m-%d"),
                "customer_id": customer_id,
                "product_id": pid,
                "qty": qty,
                "unit_price": price,
                "line_total": round(qty * price, 2),
            })
    return rows


def generate_customers(n=200):
    """Generate customers for personalisation."""
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "customer_id": i,
            "name": fake.name(),
            "phone": fake.phone_number()[:15],
            "segment": "NEW",
            "total_spend": 0.0,
            "visit_count": 0,
        })
    return rows


def seed_database(products, sales, wholesale, transactions, customers):
    conn = database.get_connection()
    try:
        # Products
        for p in products:
            conn.execute("""
                INSERT INTO products (sku, name, category, regime, mrp, wholesale_cost,
                                      base_margin_pct, shelf_life_days, unit)
                VALUES (:sku, :name, :category, :regime, :mrp, :wholesale_cost,
                        :base_margin_pct, :shelf_life_days, :unit)""", p)
        conn.commit()
        pid_map = {r["name"]: r["product_id"] for r in conn.execute("SELECT product_id, name FROM products")}

        # Re-map product_id in generated rows (they used enumeration)
        # Since products were inserted in same order, product_id is autoincrement 1..N
        # which matches our generate_daily_sales enumeration.

        conn.executemany("""
            INSERT INTO daily_sales_history (date, product_id, qty_sold, revenue, stockout)
            VALUES (:date, :product_id, :qty_sold, :revenue, :stockout)""", sales)

        conn.executemany("""
            INSERT INTO wholesale_price_history (date, product_id, cost)
            VALUES (:date, :product_id, :cost)""", wholesale)

        conn.executemany("""
            INSERT INTO customers (customer_id, name, phone, segment, total_spend, visit_count)
            VALUES (:customer_id, :name, :phone, :segment, :total_spend, :visit_count)""", customers)

        # Transactions → orders + line items
        tx_df = {}
        for r in transactions:
            tx_df.setdefault(r["transaction_id"], []).append(r)

        order_id = 0
        for tx_id, items in tx_df.items():
            order_id += 1
            cust = items[0]["customer_id"]
            date = items[0]["date"]
            total = sum(it["line_total"] for it in items)
            conn.execute("""
                INSERT INTO customer_orders (order_id, customer_id, order_date, source,
                                             status, total_value)
                VALUES (?, ?, ?, 'pos', 'COMPLETED', ?)""",
                (order_id, cust, date, total))
            for it in items:
                conn.execute("""
                    INSERT INTO order_line_items (order_id, product_id, ordered_qty,
                                                  available_qty, status, unit_price, line_total)
                    VALUES (?, ?, ?, ?, 'READY', ?, ?)""",
                    (order_id, it["product_id"], it["qty"], it["qty"],
                     it["unit_price"], it["line_total"]))

        # Initial inventory: random stock levels
        for p in products:
            stock = random.randint(20, 200) if p["regime"] == "MRP" else random.randint(50, 300)
            conn.execute("""
                INSERT INTO inventory_current (product_id, stock_on_hand, reorder_level)
                VALUES (?, ?, ?)""", (p["product_id"], stock, max(20, stock // 5)))

        conn.commit()
    finally:
        conn.close()


def run_all():
    """End-to-end data generation (idempotent — resets existing data)."""
    print("📦 Building database schema (resetting any existing data)...")
    database.build_database()
    database.reset_database()

    start_date = datetime.strptime(config.get("data.start_date"), "%Y-%m-%d")
    n_days = config.get("data.n_days")
    n_tx = config.get("data.n_transactions")

    print("🏷️  Generating products...")
    products = generate_products()
    print(f"   → {len(products)} products "
          f"({sum(1 for p in products if p['regime']=='MRP')} MRP, "
          f"{sum(1 for p in products if p['regime']=='BULK')} BULK)")

    print(f"📈 Generating {n_days} days of sales history...")
    sales = generate_daily_sales(products, start_date, n_days)

    print("💰 Generating wholesale price history...")
    wholesale = generate_wholesale_prices(products, start_date, n_days)

    print(f"🛒 Generating {n_tx} transactions...")
    transactions = generate_transactions(products, None, n_tx)

    print("👤 Generating customers...")
    customers = generate_customers(200)

    print("💾 Seeding database...")
    seed_database(products, sales, wholesale, transactions, customers)

    database.load_config_into_db()

    # Summary
    conn = database.get_connection()
    try:
        for table in ["products", "daily_sales_history", "wholesale_price_history",
                      "customer_orders", "order_line_items", "customers",
                      "inventory_current"]:
            n = database.table_count(conn, table)
            print(f"   {table:30s}: {n:>6,} rows")
    finally:
        conn.close()

    print(f"✅ Done. Database at: {config.DB_PATH}")


if __name__ == "__main__":
    run_all()
