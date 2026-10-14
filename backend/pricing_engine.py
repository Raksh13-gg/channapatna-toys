"""
Pricing and Machine Learning Engine for Channapatna Toys
Integrates:
1. Production Random Forest Regressor (Price-to-MRP ratio model from price_model.pkl)
2. TF-IDF Vocabulary Vectorizer (tfidf_vectorizer.pkl)
3. 100-Feature Schema Alignment (feature_columns.json)
4. Domain-Verified Artisan Fair-Price Calculator & Cost Breakdown
5. Marketplace vs Artisan Comparative Intelligence & Fairness Analysis
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Base directory where model artifacts are located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "price_model.pkl")
TFIDF_PATH = os.path.join(PROJECT_ROOT, "tfidf_vectorizer.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "feature_columns.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "mlr_metrics.json")

# Domain medians calculated from the production dataset
DEFAULT_MEDIANS = {
    "length_cm": 9.75,
    "width_cm": 12.00,
    "height_cm": 10.00,
    "weight_g": 75.00,
    "set_quantity": 1.00,
    "age_group": 12.00  # 12 months
}

# Standard Channapatna Craft Economics Constants
# Wood types and baseline raw cost per cubic centimeter (INR / cm3)
WOOD_COST_RATES = {
    "hale wood": 0.045,            # Wrightia tinctoria (traditional ivory-wood, sustainably harvested)
    "sustainably sourced hale wood": 0.050,
    "solid wood": 0.040,
    "pine": 0.035,
    "mango wood": 0.038,
    "beech wood": 0.048,
    "natural oak wood": 0.055,
    "alder wood": 0.045,
    "wood": 0.042,
    "other": 0.040
}

# Natural Lacquer & Vegetable Dye Cost Rate per gram of toy weight (INR / g)
# (Non-toxic natural lac + turmeric, kumkum, indigo, acacia resin)
NATURAL_LACQUER_RATE = 0.45

# Fair Living Wage benchmark for Karnataka craft artisans (INR per hour)
# Recommended standard: Rs. 120 - 150/hour (Rs. 960 - 1200/day for skilled lathe turning)
DEFAULT_FAIR_HOURLY_WAGE = 140.0


class PricingEngine:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.feature_columns = []
        self.metrics = {}
        self.is_loaded = False
        self.load_error = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            if not os.path.exists(TFIDF_PATH):
                raise FileNotFoundError(f"TF-IDF vectorizer not found at: {TFIDF_PATH}")
            if not os.path.exists(FEATURE_COLUMNS_PATH):
                raise FileNotFoundError(f"Feature columns schema not found at: {FEATURE_COLUMNS_PATH}")

            self.model = joblib.load(MODEL_PATH)
            self.tfidf = joblib.load(TFIDF_PATH)

            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {
                    "model": "RandomForestRegressor_RatioModel",
                    "mae": 194.43,
                    "rmse": 377.92,
                    "r2_score": 0.9225,
                    "mape_percent": 15.30
                }

            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)

    def get_supported_categories(self) -> list:
        """Returns clean category list supported by model one-hot encoding."""
        categories = set()
        for col in self.feature_columns:
            if col.startswith("product_type_"):
                categories.add(col.replace("product_type_", ""))
        return sorted(list(categories))

    def get_supported_materials(self) -> list:
        """Returns clean material list supported by model one-hot encoding."""
        materials = set()
        for col in self.feature_columns:
            if col.startswith("material_"):
                mat_name = col.replace("material_", "")
                # Only keep short readable materials
                if len(mat_name) < 40:
                    materials.add(mat_name)
        # Ensure traditional Channapatna wood is prominent
        preferred = ["hale wood", "sustainably sourced hale wood", "wood", "solid wood", "pine", "mango wood", "beech wood", "natural oak wood"]
        combined = [m for m in preferred if m in materials] + [m for m in sorted(list(materials)) if m not in preferred]
        return combined

    def build_feature_vector(
        self,
        product_name: str,
        length_cm: Optional[float],
        width_cm: Optional[float],
        height_cm: Optional[float],
        weight_g: Optional[float],
        set_quantity: Optional[float],
        age_group: Optional[float],
        mrp: float,
        product_type: str,
        material: str
    ) -> pd.DataFrame:
        """Constructs the exact 100-feature vector expected by the model."""
        row: Dict[str, float] = {col: 0.0 for col in self.feature_columns}

        # 1. Physical dimensions & imputation flags
        l_missing = 1.0 if (length_cm is None or length_cm <= 0) else 0.0
        w_missing = 1.0 if (width_cm is None or width_cm <= 0) else 0.0
        h_missing = 1.0 if (height_cm is None or height_cm <= 0) else 0.0
        wt_missing = 1.0 if (weight_g is None or weight_g <= 0) else 0.0
        sq_missing = 1.0 if (set_quantity is None or set_quantity <= 0) else 0.0

        actual_l = float(length_cm) if (length_cm and length_cm > 0) else DEFAULT_MEDIANS["length_cm"]
        actual_w = float(width_cm) if (width_cm and width_cm > 0) else DEFAULT_MEDIANS["width_cm"]
        actual_h = float(height_cm) if (height_cm and height_cm > 0) else DEFAULT_MEDIANS["height_cm"]
        actual_wt = float(weight_g) if (weight_g and weight_g > 0) else DEFAULT_MEDIANS["weight_g"]
        actual_sq = float(set_quantity) if (set_quantity and set_quantity > 0) else DEFAULT_MEDIANS["set_quantity"]
        actual_age = float(age_group) if (age_group and age_group > 0) else DEFAULT_MEDIANS["age_group"]

        row["length_cm"] = actual_l
        row["width_cm"] = actual_w
        row["height_cm"] = actual_h
        row["weight_g"] = actual_wt
        row["set_quantity"] = actual_sq
        row["age_group"] = actual_age

        row["length_cm_missing"] = l_missing
        row["width_cm_missing"] = w_missing
        row["height_cm_missing"] = h_missing
        row["weight_g_missing"] = wt_missing
        row["set_quantity_missing"] = sq_missing

        # 2. Geometric calculations
        volume = actual_l * actual_w * actual_h
        density = actual_wt / (volume + 0.001)
        row["volume_cm3"] = volume
        row["density_g_cm3"] = density

        # 3. Reference MRP and Log-transform
        clean_mrp = max(10.0, float(mrp))
        row["mrp"] = clean_mrp
        row["log_mrp"] = float(np.log1p(clean_mrp))

        # 4. Text TF-IDF features
        clean_name = str(product_name or "").strip()
        if self.tfidf is not None and clean_name:
            tfidf_vec = self.tfidf.transform([clean_name]).toarray()[0]
            for term, val in zip(self.tfidf.get_feature_names_out(), tfidf_vec):
                col_name = f"word_{term}"
                if col_name in row:
                    row[col_name] = float(val)

        # 5. One-hot product_type
        clean_type = str(product_type or "").lower().strip()
        type_col = f"product_type_{clean_type}"
        if type_col in row:
            row[type_col] = 1.0
        elif "spinning" in clean_type or "top" in clean_type:
            if "product_type_traditional game" in row:
                row["product_type_traditional game"] = 1.0
            elif "product_type_wooden toy" in row:
                row["product_type_wooden toy"] = 1.0
        elif "peg" in clean_type or "doll" in clean_type:
            if "product_type_nesting doll toy" in row:
                row["product_type_nesting doll toy"] = 1.0
            elif "product_type_doll" in row:
                row["product_type_doll"] = 1.0

        # 6. One-hot material
        clean_mat = str(material or "").lower().strip()
        mat_col = f"material_{clean_mat}"
        if mat_col in row:
            row[mat_col] = 1.0
        elif f"material_wood" in row and "wood" in clean_mat:
            row["material_wood"] = 1.0

        df_row = pd.DataFrame([row])[self.feature_columns].astype(float)
        return df_row

    def predict_marketplace_price(
        self,
        product_name: str,
        length_cm: Optional[float],
        width_cm: Optional[float],
        height_cm: Optional[float],
        weight_g: Optional[float],
        set_quantity: Optional[float],
        age_group: Optional[float],
        mrp: float,
        product_type: str,
        material: str
    ) -> Dict[str, Any]:
        """Predicts the marketplace selling price using the Price-to-MRP ratio model."""
        if not self.is_loaded:
            raise RuntimeError(f"Model artifacts not loaded: {self.load_error}")

        if mrp <= 0:
            raise ValueError("Reference MRP must be greater than zero.")

        X = self.build_feature_vector(
            product_name=product_name,
            length_cm=length_cm,
            width_cm=width_cm,
            height_cm=height_cm,
            weight_g=weight_g,
            set_quantity=set_quantity,
            age_group=age_group,
            mrp=mrp,
            product_type=product_type,
            material=material
        )

        raw_ratio = float(self.model.predict(X)[0])

        # Physical and craft sensitivity adjustments:
        # 1. Volumetric scale adjustment (scaled smoothly across small to large toys)
        volume = float(X["volume_cm3"].values[0])
        med_volume = 1170.0  # reference median volume cm3
        vol_factor = float(np.clip((volume / med_volume) ** 0.07, 0.82, 1.15))

        # 2. Weight and density adjustment: heavier pieces convey craftsmanship and wood quality
        weight = float(X["weight_g"].values[0])
        density = float(X["density_g_cm3"].values[0])
        med_weight = 200.0  # reference median weight g
        wt_factor = float(np.clip((weight / med_weight) ** 0.05, 0.85, 1.12))

        # 3. Material quality grade factor
        mat_map = {
            "sustainably sourced hale wood": 1.04,
            "hale wood": 1.02,
            "solid wood": 0.99,
            "wood": 0.97,
            "mango wood": 0.95,
            "pine": 0.92
        }
        clean_mat_name = str(material or "").lower().strip()
        mat_factor = mat_map.get(clean_mat_name, 0.98)

        # 4. Product type / Category market tier factor
        cat_map = {
            "rocking toy": 1.03,
            "nesting dolls": 1.02,
            "peg dolls": 1.02,
            "educational toy": 1.00,
            "stacking toy": 0.98,
            "wooden toy": 0.97,
            "puzzle": 0.95,
            "spinning top": 0.93,
            "baby rattle": 0.90
        }
        clean_cat_name = str(product_type or "").lower().strip()
        cat_factor = cat_map.get(clean_cat_name, 0.97)

        # 5. Set quantity bundle adjustment
        actual_qty = float(X["set_quantity"].values[0])
        qty_factor = float(np.clip(1.0 + (actual_qty - 1.0) * 0.015, 0.95, 1.15))

        # Composite physics multiplier normalized around baseline
        composite_mult = (vol_factor * wt_factor * mat_factor * cat_factor * qty_factor) / 1.0
        
        # Grounded prediction: base ML ratio scaled by composite physics and craft characteristics
        # Calibrated ratio smoothly scales without premature clipping saturation
        calibrated_ratio = float(np.clip(raw_ratio * 0.91 * composite_mult, 0.40, 0.985))
        predicted_price = round(calibrated_ratio * mrp, 2)
        implied_discount_percent = round((1.0 - calibrated_ratio) * 100, 1)

        # Estimated confidence range based on production RMSE (Rs. 377.92) or MAPE (15.3%)
        mape = self.metrics.get("mape_percent", 15.30) / 100.0
        lower_bound = max(round(predicted_price * (1 - mape), 2), 25.0)
        upper_bound = min(round(predicted_price * (1 + mape), 2), round(mrp * 1.05, 2))

        return {
            "predicted_price_inr": predicted_price,
            "predicted_ratio": round(calibrated_ratio, 4),
            "reference_mrp": float(mrp),
            "implied_discount_percent": implied_discount_percent,
            "price_range_low": lower_bound,
            "price_range_high": upper_bound,
            "model_used": self.metrics.get("model", "RandomForestRegressor_RatioModel"),
            "features_utilized": len(self.feature_columns),
            "input_summary": {
                "product_name": product_name,
                "product_type": product_type,
                "material": material,
                "volume_cm3": round(volume, 2),
                "density_g_cm3": round(density, 4),
                "set_quantity": int(actual_qty)
            }
        }

    def calculate_artisan_fair_price(
        self,
        length_cm: Optional[float],
        width_cm: Optional[float],
        height_cm: Optional[float],
        weight_g: Optional[float],
        material: str,
        set_quantity: int = 1,
        crafting_hours: Optional[float] = None,
        hourly_wage: float = DEFAULT_FAIR_HOURLY_WAGE,
        artisan_margin_percent: float = 25.0
    ) -> Dict[str, Any]:
        """
        Calculates authentic artisan fair price and transparent cost breakdown.
        Based on Channapatna GI craft standards:
        - Wood cost (Hale mara / Wrightia tinctoria)
        - Natural vegetable lacquer & natural dyes
        - Skilled lathe turning and hand-carving labor
        - Workshop tooling & natural pine polish leaf overhead
        - Fair artisan margin (20-30%)
        """
        l = float(length_cm) if (length_cm and length_cm > 0) else DEFAULT_MEDIANS["length_cm"]
        w = float(width_cm) if (width_cm and width_cm > 0) else DEFAULT_MEDIANS["width_cm"]
        h = float(height_cm) if (height_cm and height_cm > 0) else DEFAULT_MEDIANS["height_cm"]
        wt = float(weight_g) if (weight_g and weight_g > 0) else DEFAULT_MEDIANS["weight_g"]
        qty = max(1, int(set_quantity or 1))

        volume = l * w * h

        # 1. Wood material cost
        clean_mat = str(material or "hale wood").lower().split("(", 1)[0].strip()
        wood_rate = WOOD_COST_RATES.get(clean_mat, WOOD_COST_RATES["wood"])
        # Add 25% wood turning lathe wastage allowance (chips & shavings during turning)
        wood_cost = round(volume * wood_rate * 1.25 * qty, 2)
        # Floor wood cost
        wood_cost = max(wood_cost, round(35.0 * qty, 2))

        # 2. Natural vegetable lac & coloring cost
        lacquer_cost = round(wt * NATURAL_LACQUER_RATE * qty, 2)
        lacquer_cost = max(lacquer_cost, round(25.0 * qty, 2))

        # 3. Labor Cost (Skilled lathe turning, carving, assembly, polishing)
        if crafting_hours is None or crafting_hours <= 0:
            # Estimate crafting hours dynamically based on physical dimensions, weight & pieces
            base_hours = 1.0
            if volume > 1500 or wt > 250:
                base_hours = 2.5
            elif volume > 800 or wt > 150:
                base_hours = 1.75
            elif volume < 200 and wt < 50:
                base_hours = 0.75

            # Multi-piece sets require proportional lathe turning time
            crafting_hours = round(base_hours * (1.0 + 0.5 * (qty - 1)), 2)

        wage_rate = max(80.0, float(hourly_wage))
        labor_cost = round(crafting_hours * wage_rate, 2)

        # 4. Lathe maintenance, sandpapers, screw-pine (talagari) polishing leaves & workshop overhead (~15% of material+labor)
        overhead_cost = round((wood_cost + lacquer_cost + labor_cost) * 0.15, 2)

        # 5. Direct Artisan Production Cost
        base_production_cost = round(wood_cost + lacquer_cost + labor_cost + overhead_cost, 2)

        # 6. Fair Artisan Profit Margin (Guarantees living wage security & craft continuation)
        margin_pct = max(10.0, min(50.0, float(artisan_margin_percent)))
        artisan_margin = round(base_production_cost * (margin_pct / 100.0), 2)

        # Final Fair Artisan Price
        fair_price = round(base_production_cost + artisan_margin, 2)

        return {
            "fair_artisan_price_inr": fair_price,
            "cost_breakdown": {
                "wood_raw_material_inr": wood_cost,
                "natural_lacquer_and_dyes_inr": lacquer_cost,
                "artisan_labor_inr": labor_cost,
                "workshop_overhead_tools_inr": overhead_cost,
                "fair_artisan_margin_inr": artisan_margin
            },
            "parameters": {
                "crafting_hours": crafting_hours,
                "hourly_wage_inr": wage_rate,
                "artisan_margin_percent": margin_pct,
                "volume_cm3": round(volume, 2),
                "weight_g": round(wt, 2),
                "set_quantity": qty,
                "wood_type": clean_mat,
                "wood_cost_rate_inr_per_cm3": wood_rate,
                "wood_wastage_allowance_percent": 25.0,
                "natural_lacquer_rate_inr_per_g": NATURAL_LACQUER_RATE,
                "workshop_overhead_percent": 15.0
            },
            "base_production_cost_inr": base_production_cost
        }

    def analyze_fairness_and_market(
        self,
        predicted_marketplace_price: float,
        fair_artisan_price: float,
        reference_mrp: float
    ) -> Dict[str, Any]:
        """
        Compares predicted marketplace price vs artisan fair price and provides
        rich market intelligence explanations and actionable insights.
        """
        diff = round(predicted_marketplace_price - fair_artisan_price, 2)
        markup_ratio = round(predicted_marketplace_price / max(1.0, fair_artisan_price), 2)
        artisan_share_pct = round((fair_artisan_price / max(1.0, predicted_marketplace_price)) * 100, 1)

        # Determine fairness status
        if diff < -50:
            status = "EXPLOITATION_RISK"
            status_label = "Underpriced (Artisan Living Wage at Risk)"
            theme = "danger"
            explanation = (
                f"The predicted marketplace selling price (₹{predicted_marketplace_price}) is ₹{abs(diff):.2f} "
                f"LOWER than the artisan's fair living cost (₹{fair_artisan_price}). "
                "Selling at this rate forces artisans to compromise on fair wages or use cheaper synthetic paints "
                "instead of genuine Channapatna vegetable lacquer. Consider adjusting the base price upward to safeguard artisan livelihood."
            )
            recommendation = (
                f"Recommended floor selling price: ₹{fair_artisan_price:.2f}. "
                "Highlight the GI-tag authenticity, Hale wood sustainability, and hand-lathe heritage to justify fair retail valuation."
            )
        elif abs(diff) <= 50:
            status = "BALANCED_FAIR"
            status_label = "Optimal Fair Price Balance"
            theme = "success"
            explanation = (
                f"The predicted marketplace price (₹{predicted_marketplace_price}) closely aligns with the "
                f"artisan fair compensation threshold (₹{fair_artisan_price}). "
                f"Artisan receives an equitable {artisan_share_pct}% share of the retail consumer price, ensuring both fair trade sustainability and market competitiveness."
            )
            recommendation = (
                "Maintain this pricing corridor. It offers fair remuneration while remaining attractive to conscious handicraft buyers."
            )
        elif markup_ratio <= 1.6:
            status = "MODERATE_RETAIL_MARKUP"
            status_label = "Healthy Marketplace Corridor"
            theme = "info"
            explanation = (
                f"Marketplace price (₹{predicted_marketplace_price}) provides a healthy {round((markup_ratio - 1) * 100)}% commercial margin over artisan fair cost (₹{fair_artisan_price}). "
                "This spread accommodates typical e-commerce platform fees (12-18%), safe packaging, and logistics while preserving the artisan's full living wage."
            )
            recommendation = (
                "Ensure that platform commission does not exceed 20% to prevent squeezing artisan margins during festive promotional discounts."
            )
        else:
            status = "HIGH_INTERMEDIARY_MARKUP"
            status_label = "High Intermediary / Premium Markup"
            theme = "warning"
            explanation = (
                f"Marketplace price (₹{predicted_marketplace_price}) is {markup_ratio}x the artisan fair price (₹{fair_artisan_price}). "
                f"Artisan receives only {artisan_share_pct}% of the retail price. "
                "Significant margin is captured by middlemen, brand markups, or marketplace distribution channels."
            )
            recommendation = (
                "Consider direct artisan-to-consumer (D2C) channels or cooperative pricing. "
                f"Passing even 15% of the intermediary markup back to the artisan increases their family income by ₹{round(diff * 0.15, 2)} per unit."
            )

        return {
            "status": status,
            "status_label": status_label,
            "theme": theme,
            "price_difference_inr": diff,
            "markup_ratio": markup_ratio,
            "artisan_share_percent": min(100.0, artisan_share_pct),
            "marketplace_share_percent": max(0.0, round(100.0 - artisan_share_pct, 1)),
            "explanation": explanation,
            "recommendation": recommendation
        }


# Global engine singleton
pricing_engine = PricingEngine()
