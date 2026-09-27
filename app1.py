from flask import Flask, render_template, request, jsonify

import os
import numpy as np
import pandas as pd
import joblib
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot import chatbot
from voice_bot import process_voice

try:
    from ai.automation import (
        generate_risk_report,
        identify_high_risk_orders,
        logistics_orchestrator
    )
except Exception as e:
    print("AI automation module could not be loaded:", e)

    generate_risk_report = None
    identify_high_risk_orders = None
    logistics_orchestrator = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "DataCoSupplyChainDataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "late_delivery_model.pkl"
)

print("\nLoading DataCo dataset...")

try:

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    print(
        "Dataset loaded successfully:",
        df.shape
    )

except Exception as e:

    print(
        "Dataset could not be loaded:",
        e
    )

    df = pd.DataFrame()


delivery_model = None
model_features = []

try:
    print("\nLoading delivery-risk model...")

    model_package = joblib.load(
        MODEL_PATH
    )

    delivery_model = model_package["model"]

    model_features = model_package["features"]

    print(
        "Delivery model loaded successfully."
    )

    print(
        "Number of model features:",
        len(model_features)
    )

except Exception as e:
    print("Delivery model could not be loaded. Reason:", e)



print("\nPreparing AI Search...")

try:
    product_columns = [
        "Product Name",
        "Department Name",
        "Category Name",
        "Product Price",
        "Sales",
        "Order Item Quantity",
        "Late_delivery_risk"
    ]

    products = df[product_columns].copy()

    products = (products.groupby("Product Name", as_index=False)
    .agg({
        "Department Name": "first",
        "Category Name": "first",
        "Product Price": "mean",
        "Sales": "sum",
        "Order Item Quantity": "sum",
        "Late_delivery_risk": "mean"
    })
    )
    
    products = products.reset_index(drop=True)

    products["search_text"] = (
        products["Product Name"].fillna("").astype(str)
        + " "
        + products["Department Name"].fillna("").astype(str)
        + " "
        + products["Category Name"].fillna("").astype(str)
    )

    print(
        "Unique products:",
        len(products)
    )


    search_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    product_embeddings = search_model.encode(
        products["search_text"].tolist(),
        show_progress_bar=False
    )

    print(
        "AI Search loaded successfully."
    )

    print(
        "Embedding shape:",
        product_embeddings.shape
    )

except Exception as e:
    print(
        "AI Search could not be loaded:",
        e
    )

    products = pd.DataFrame()
    search_model = None
    product_embeddings = None


@app.route("/")
def home():
    return render_template(
        "index.html"
    )

@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html"
    )


@app.route("/prediction")
def prediction():
    return render_template(
        "prediction.html"
    )


@app.route("/search")
def search_page():
    return render_template(
        "search.html"
    )

@app.route("/api/search", methods=["GET", "POST"])
def ai_search():
    if request.method == "GET":
        query = request.args.get(
            "q",
            ""
        ).strip()

    else:
        data = request.get_json(
            silent=True
        ) or {}

        query = data.get(
            "query",
            ""
        ).strip()

    if not query:

        return jsonify({
            "error": "Please enter a search query."
        }), 400

    try:
        query_lower = query.lower()

        filtered_products = products.copy()

        max_price = None

        price_match = re.search(
            r"(?:under|below|less\s+than)\s*\$?\s*(\d+(?:\.\d+)?)",
            query_lower
        )

        if price_match:

            max_price = float(
                price_match.group(1)
            )

        if max_price is None:

            price_match = re.search(
                r"\$?\s*(\d+(?:\.\d+)?)\s*(?:less\s+cost|less\s+expensive|dollars?)",
                query_lower
            )

            if price_match:

                max_price = float(
                    price_match.group(1)
                )

        if max_price is not None:

            filtered_products = filtered_products[
                pd.to_numeric(
                    filtered_products["Product Price"],
                    errors="coerce"
                ) <= max_price
            ]

        low_risk_keywords = [
            "low delivery risk",
            "low risk",
            "arrive on time",
            "arrive on time products",
            "likely to arrive on time",
            "likely arrive on time",
            "on time",
            "safe delivery",
            "low chance of late delivery",
            "unlikely to be late",
            "unlikely late",
            "less likely to be late"
            ]

        high_risk_keywords = [
            "likely to be late",
            "late delivery",
            "high delivery risk",
            "high risk",
            "likely late",
            "late products",
            "high chance of late delivery"
            ]

        low_risk_requested = any(
                    keyword in query_lower
                    for keyword in low_risk_keywords
                )

        high_risk_requested = any(
            keyword in query_lower
            for keyword in high_risk_keywords
        )

        if low_risk_requested:
            filtered_products = filtered_products[
                filtered_products["Late_delivery_risk"] <= 0.50
            ]

        elif high_risk_requested:
            filtered_products = filtered_products[
                filtered_products["Late_delivery_risk"] >= 0.70
            ]

        if "sporting goods" in query_lower:
            sporting_text = (
                filtered_products["Product Name"]
                .fillna("")
                .astype(str)
                + " "
                + filtered_products["Department Name"]
                .fillna("")
                .astype(str)
                + " "
                + filtered_products["Category Name"]
                .fillna("")
                .astype(str)
                ).str.lower()
            
            sporting_keywords = (
                "golf|basketball|football|soccer|"
                "baseball|tennis|running|cycling|"
                "fitness|hiking|boxing|mma|"
                "sport|athletic|training"
                )
            
            sporting_mask = sporting_text.str.contains(
                sporting_keywords,
                regex=True,
                na=False
                )
            
            filtered_products = filtered_products[
                sporting_mask
                ]
            
        if "golf" in query_lower:
            golf_mask = (
                filtered_products["Product Name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains("golf", na=False)
                    
                |
                    
                filtered_products["Department Name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains("golf", na=False)
                
                |
                    
                filtered_products["Category Name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains("golf", na=False)
                )
            
            filtered_products = filtered_products[
                golf_mask
                ]
                
        if filtered_products.empty:
            return jsonify({
                "query": query,
                "results": [],
                "message": (
                "No products matched the "
                "requested conditions."
                )
                })
                    
        search_text = (
            filtered_products["Product Name"]
                .fillna("")
                .astype(str)
                + " "
                + filtered_products["Department Name"]
                .fillna("")
                .astype(str)
                + " "
                + filtered_products["Category Name"]
                .fillna("")
                .astype(str)        
                )

        query_embedding = search_model.encode(
                    [query],        
                    show_progress_bar=False
                )

        filtered_embeddings = search_model.encode(
            search_text.tolist(),
            show_progress_bar=False
        )

        similarities = cosine_similarity(
            query_embedding,
            filtered_embeddings
        )[0]

        filtered_products = filtered_products.copy()

        filtered_products["similarity_score"] = similarities

        if low_risk_requested:
            filtered_products["final_score"] = (
                filtered_products["similarity_score"] * 0.70
                +
                (1 - filtered_products["Late_delivery_risk"])
                * 0.30
            )

        elif high_risk_requested:
            filtered_products["final_score"] = (
                filtered_products["similarity_score"] * 0.70
                +
                filtered_products["Late_delivery_risk"]
                * 0.30
            )

        else:
            filtered_products["final_score"] = (
                filtered_products["similarity_score"]
            )

        cheap_keywords = [
            "cheap",
            "affordable",
            "budget",
            "low cost",
            "inexpensive",
            "cheapest"
        ]

        cheap_requested = any(
            keyword in query_lower
            for keyword in cheap_keywords
        )

        if cheap_requested:
            prices = pd.to_numeric(
                filtered_products["Product Price"],
                errors="coerce"
            )

            max_product_price = prices.max()

            if (pd.notna(max_product_price) and max_product_price > 0):
                price_score = (
                    1 -
                    (
                        prices /
                        max_product_price
                    )
                )

                filtered_products["final_score"] += (
                    price_score.fillna(0) * 0.20
                )

        filtered_products = filtered_products.sort_values(
            "final_score",
            ascending=False
        )

        top_products = filtered_products.head(10)

        results = []

        for _, product in top_products.iterrows():

            price = pd.to_numeric(
                product["Product Price"],
                errors="coerce"
            )

            sales = pd.to_numeric(
                product["Sales"],
                errors="coerce"
            )

            quantity = pd.to_numeric(
                product["Order Item Quantity"],
                errors="coerce"
            )

            risk = pd.to_numeric(
                product["Late_delivery_risk"],
                errors="coerce"
            )

            risk_percentage = (
                float(risk) * 100
                if pd.notna(risk)
                else None
            )

            results.append({
                "product_name": str(
                    product["Product Name"]
                ),

                "department": str(
                    product["Department Name"]
                ),

                "category": str(
                    product["Category Name"]
                ),

                "price": (
                    float(price)
                    if pd.notna(price)
                    else 0.0
                ),

                "sales": (
                    float(sales)
                    if pd.notna(sales)
                    else 0.0
                ),

                "quantity": (
                    int(quantity)
                    if pd.notna(quantity)
                    else 0
                ),

                "delivery_risk": risk_percentage,

                "late_delivery_risk": risk_percentage,

                "similarity": round(
                    float(
                        product["similarity_score"]
                    ) * 100,
                    2
                ),

                "final_score": round(
                    float(
                        product["final_score"]
                    ) * 100,
                    2
                )
            })

        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        print("AI Search error:", e)
        return jsonify({
            "error": str(e)
        }), 500

@app.route(
    "/api/predict-risk",
    methods=["POST"]
)
def predict_risk():

    if delivery_model is None:

        return jsonify({
            "error": "Delivery prediction model is not available."
        }), 500

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "No prediction data received."
        }), 400

    try:
        input_data = pd.DataFrame([data])

        numeric_columns = [
            "Order Item Discount Rate",
            "Sales",
            "Order Item Product Price",
            "Order Item Quantity"
        ]

        for column in numeric_columns:

            if column in input_data.columns:

                input_data[column] = pd.to_numeric(
                    input_data[column],
                    errors="coerce"
                )

        for column in input_data.columns:

            if (
                input_data[column].dtype == "object"
                or pd.api.types.is_string_dtype(
                    input_data[column]
                )
            ):

                input_data[column] = (
                    input_data[column]
                    .fillna("Unknown")
                )

            else:

                input_data[column] = (
                    input_data[column]
                    .fillna(0)
                )

        categorical_columns = [
            "Shipping Mode",
            "Market",
            "Order Region",
            "Department Name",
            "Category Name",
            "Customer Country"
        ]

        categorical_columns = [
            column
            for column in categorical_columns
            if column in input_data.columns
        ]

        input_data = pd.get_dummies(
            input_data,
            columns=categorical_columns,
            drop_first=True
        )


        input_data = input_data.reindex(
            columns=model_features,
            fill_value=0
        )

        prediction = delivery_model.predict(
            input_data
        )[0]

        try:

            probability = delivery_model.predict_proba(
                input_data
            )[0][1]

        except Exception:

            probability = float(prediction)

        risk_score = round(
            float(probability) * 100,
            2
        )

        if risk_score >= 80:

            risk_level = "CRITICAL"

        elif risk_score >= 60:

            risk_level = "HIGH"

        elif risk_score >= 40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        
        return jsonify({

            "prediction": int(
                prediction
            ),

            "risk_score": risk_score,

            "risk_level": risk_level,

            "message":
                "High probability of late delivery."
                if prediction == 1
                else
                "Low probability of late delivery."
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/chatbot")
def chatbot_page():

    return render_template(
        "chatbot.html"
    )

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    data = request.get_json()

    message = data.get(
        "message",
        ""
    ).strip()

    if not message:

        return jsonify({
            "error": "Please enter a message."
        }), 400

    try:

        response = chatbot(
            message
        )

        return jsonify({
            "response": response
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@app.route(
    "/api/voice",
    methods=["POST"]
)
def voice():

    try:

        audio_file = request.files.get(
            "audio"
        )

        if audio_file is None:

            return jsonify({
                "error": "No audio file received."
            }), 400

        response = process_voice(
            audio_file
        )

        return jsonify({
            "response": response
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/automation")
def automation_page():

    return render_template(
        "automation.html"
    )

@app.route(
    "/api/automation/report",
    methods=["GET"]
)
def automation_report():

    if generate_risk_report is None:

        return jsonify({
            "error":
                "AI automation module is not available."
        }), 500

    try:

        report = generate_risk_report(
            df
        )

        return jsonify(report)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/api/automation/high-risk",
    methods=["GET"]
)
def high_risk_orders():

    if identify_high_risk_orders is None:

        return jsonify({
            "error":
                "High-risk order function is not available."
        }), 500

    try:

        result = identify_high_risk_orders(
            df
        )

        if isinstance(result, pd.DataFrame):

            result = result.head(100).to_dict(
                orient="records"
            )

        return jsonify({
            "result": result
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/api/dashboard/report", methods=["GET", "POST"])
def dashboard_data():
    try:
        if request.method == "POST":

            data = request.get_json(silent=True) or {}

            market = data.get("market")
            shipping_mode = data.get("shipping_mode")

        else:
            market = request.args.get("market")
            shipping_mode = request.args.get("shipping_mode")

        dashboard_df = df.copy()

        if market:
            dashboard_df = dashboard_df[
                dashboard_df["Market"] == market
            ]

        if shipping_mode:
            dashboard_df = dashboard_df[
                dashboard_df["Shipping Mode"] == shipping_mode
            ]

        total_records = len(dashboard_df)

        high_risk_orders = int(
            (dashboard_df["Late_delivery_risk"] == 1).sum()
        )


        low_risk_orders = int(
            (dashboard_df["Late_delivery_risk"] == 0).sum()
        )

        if total_records > 0:
            risk_rate = round(
                (high_risk_orders / total_records) * 100,
                2
            )
        else:
            risk_rate = 0

        total_products = int(
            dashboard_df["Product Name"].nunique()
        )

        shipping_risk = (
            dashboard_df
            .groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        )

        market_risk = (
            dashboard_df
            .groupby("Market")["Late_delivery_risk"]
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        )

        delivery_status = (
            dashboard_df["Delivery Status"]
            .value_counts()
            .to_dict()

        )

        return jsonify({
            "total_records": total_records,
            "high_risk": high_risk_orders,
            "low_risk": low_risk_orders,
            "risk_rate": risk_rate,
            "total_products": total_products,
            "delivery_status": delivery_status,
            "shipping_risk": shipping_risk,
            "market_risk": market_risk
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )