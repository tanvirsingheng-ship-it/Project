import pandas as pd
import re

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1"
)

def percentage(value):
    return f"{value * 100:.2f}%"


def get_overall_risk():
    return df["Late_delivery_risk"].mean()

def detect_intent(message):

    message = message.lower().strip()

    if any(word in message.split() for word in [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]):
        return "greeting"

    if (
        "help" in message
        or "what can you do" in message
        or "what can you help" in message
    ):
        return "help"

    if (
        "how many orders" in message
        or "total orders" in message
        or "number of orders" in message
        or "order count" in message
    ):
        return "order_count"

    if (
        "overall risk" in message
        or "average delivery risk" in message
        or "overall delivery risk" in message
        or "average late delivery" in message
    ):
        return "overall_risk"

    if (
        "shipping mode" in message
        or "shipping method" in message
        or "delivery method" in message
    ):
        return "shipping_mode"

    if (
        "region" in message
        or "regions" in message
        or "area" in message
        or "areas" in message
    ):
        return "region"

    if (
        "market" in message
        or "markets" in message
    ):
        return "market"

    if (
        "category" in message
        or "categories" in message
    ):
        return "category"

    if (
        "product" in message
        or "products" in message
    ):
        return "product"

    if (
        "shipping days" in message
        or "shipping duration" in message
        or "delivery duration" in message
        or "days for shipping" in message
    ):
        return "shipping_days"

    if (
        "why" in message
        and (
            "late" in message
            or "delay" in message
        )
    ):
        return "why_late"


    return "unknown"

def greeting_response():

    return (
        "Hello! I'm your Supply Chain AI Assistant. "
        "I can analyze late-delivery risk, shipping modes, "
        "regions, markets, categories, products, and order data."
    )


def help_response():

    return (
        "I can help you with:\n\n"
        "• Overall late-delivery risk\n"
        "• Shipping-mode risk\n"
        "• Region and market analysis\n"
        "• Category-level delivery risk\n"
        "• Product-level risk\n"
        "• Order counts\n"
        "• Shipping-duration analysis\n\n"
        "Example: \"Which shipping mode has the highest risk?\""
    )


def order_count_response():

    total_orders = len(df)

    return (
        f"The dataset contains {total_orders:,} "
        f"order records."
    )

def overall_risk_response():

    risk = get_overall_risk()

    return (
        f"The overall historical late-delivery risk "
        f"is {percentage(risk)} across {len(df):,} "
        f"order records."
    )

def shipping_mode_response(message):

    risk = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_mode = risk.index[0]
    highest_value = risk.iloc[0]

    lowest_mode = risk.index[-1]
    lowest_value = risk.iloc[-1]


    if any(word in message for word in [
        "highest",
        "maximum",
        "most",
        "riskiest",
        "worst"
    ]):

        return (
            f"{highest_mode} has the highest historical "
            f"late-delivery risk at {percentage(highest_value)}.\n\n"
            f"The lowest is {lowest_mode} at "
            f"{percentage(lowest_value)}."
        )

    if any(word in message for word in [
        "lowest",
        "minimum",
        "least",
        "safest",
        "best"
    ]):

        return (
            f"{lowest_mode} has the lowest historical "
            f"late-delivery risk at {percentage(lowest_value)}.\n\n"
            f"The highest is {highest_mode} at "
            f"{percentage(highest_value)}."
        )

    lines = []

    for mode, value in risk.items():

        lines.append(
            f"• {mode}: {percentage(value)}"
        )

    return (
        "Historical late-delivery risk by shipping mode:\n\n"
        + "\n".join(lines)
    )

def region_response(message):

    risk = (
        df.groupby("Order Region")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_region = risk.index[0]
    highest_value = risk.iloc[0]

    lowest_region = risk.index[-1]
    lowest_value = risk.iloc[-1]


    if any(word in message for word in [
        "highest",
        "maximum",
        "most",
        "riskiest",
        "worst"
    ]):

        return (
            f"{highest_region} has the highest historical "
            f"late-delivery risk at {percentage(highest_value)}.\n\n"
            f"{lowest_region} has the lowest at "
            f"{percentage(lowest_value)}."
        )


    if any(word in message for word in [
        "lowest",
        "minimum",
        "least",
        "safest"
    ]):

        return (
            f"{lowest_region} has the lowest historical "
            f"late-delivery risk at {percentage(lowest_value)}.\n\n"
            f"{highest_region} has the highest at "
            f"{percentage(highest_value)}."
        )


    lines = []

    for region, value in risk.head(10).items():

        lines.append(
            f"• {region}: {percentage(value)}"
        )

    return (
        "Top regions by historical late-delivery risk:\n\n"
        + "\n".join(lines)
    )


def market_response(message):

    risk = (
        df.groupby("Market")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_market = risk.index[0]
    highest_value = risk.iloc[0]

    lowest_market = risk.index[-1]
    lowest_value = risk.iloc[-1]


    if "highest" in message or "riskiest" in message:

        return (
            f"{highest_market} has the highest historical "
            f"late-delivery risk among the markets at "
            f"{percentage(highest_value)}."
        )


    if "lowest" in message or "safest" in message:

        return (
            f"{lowest_market} has the lowest historical "
            f"late-delivery risk at "
            f"{percentage(lowest_value)}."
        )


    lines = []

    for market, value in risk.items():

        lines.append(
            f"• {market}: {percentage(value)}"
        )

    return (
        "Historical late-delivery risk by market:\n\n"
        + "\n".join(lines)
    )

def category_response(message):

    risk = (
        df.groupby("Category Name")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_category = risk.index[0]
    highest_value = risk.iloc[0]

    lowest_category = risk.index[-1]
    lowest_value = risk.iloc[-1]


    if "highest" in message or "riskiest" in message:

        return (
            f"{highest_category} has the highest historical "
            f"late-delivery risk at "
            f"{percentage(highest_value)}."
        )


    if "lowest" in message or "safest" in message:

        return (
            f"{lowest_category} has the lowest historical "
            f"late-delivery risk at "
            f"{percentage(lowest_value)}."
        )


    lines = []

    for category, value in risk.head(10).items():

        lines.append(
            f"• {category}: {percentage(value)}"
        )

    return (
        "Top categories by historical late-delivery risk:\n\n"
        + "\n".join(lines)
    )

def shipping_days_response():

    risk = (
        df.groupby("Late_delivery_risk")[
            [
                "Days for shipping (real)",
                "Days for shipment (scheduled)"
            ]
        ]
        .mean()
    )

    if 0 in risk.index and 1 in risk.index:

        normal_real = risk.loc[0, "Days for shipping (real)"]
        late_real = risk.loc[1, "Days for shipping (real)"]

        normal_scheduled = risk.loc[
            0,
            "Days for shipment (scheduled)"
        ]

        late_scheduled = risk.loc[
            1,
            "Days for shipment (scheduled)"
        ]

        return (
            "Average shipping duration comparison:\n\n"
            f"• No late risk — actual: {normal_real:.2f} days, "
            f"scheduled: {normal_scheduled:.2f} days\n"
            f"• Late risk — actual: {late_real:.2f} days, "
            f"scheduled: {late_scheduled:.2f} days\n\n"
            "Late-risk orders therefore show a larger gap "
            "between actual and scheduled shipping duration."
        )

    return "Shipping-duration information is unavailable."


def why_late_response():

    shipping = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    mode = shipping.index[0]
    mode_risk = shipping.iloc[0]

    duration = (
        df.groupby("Late_delivery_risk")[
            "Days for shipping (real)"
        ].mean()
    )

    explanation = (
        f"One major pattern in the data is shipping-mode risk: "
        f"{mode} has the highest historical late-delivery risk "
        f"at {percentage(mode_risk)}.\n\n"
    )

    if 0 in duration.index and 1 in duration.index:

        explanation += (
            f"Late-risk orders also average "
            f"{duration.loc[1]:.2f} actual shipping days, "
            f"compared with {duration.loc[0]:.2f} days for "
            f"orders without late risk."
        )

    return explanation


def product_response(message):
    stop_words = {
        "show",
        "find",
        "me",
        "products",
        "product",
        "with",
        "low",
        "high",
        "delivery",
        "risk",
        "late",
        "the",
        "a",
        "an",
        "and",
        "for",
        "of",
        "in"
    }

    words = re.findall(
        r"[a-zA-Z0-9]+",
        message.lower()
    )

    keywords = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]


    if not keywords:
        return (
            "Please provide a product name, category, "
            "or search description."
        )


    pattern = "|".join(
        re.escape(word)
        for word in keywords
    )


    matches = df[
        df["Product Name"]
        .fillna("")
        .str.lower()
        .str.contains(
            pattern,
            regex=True
        )
    ]


    if matches.empty:
        return (
            "I couldn't find a matching product in "
            "the dataset."
        )


    product_risk = (
        matches.groupby("Product Name")[
            "Late_delivery_risk"
        ]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )


    lines = []

    for product, value in product_risk.items():
        lines.append(
            f"• {product}: {percentage(value)} late-delivery risk"
        )

    return (
        "Matching products and their historical "
        "late-delivery risk:\n\n"
        + "\n".join(lines)
    )

def chatbot(message):

    if not isinstance(message, str):

        return (
            "I couldn't process that message. "
            "Please enter your question as text."
        )


    message = message.strip()

    if not message:
        return "Please enter a question."

    intent = detect_intent(message)

    if intent == "greeting":
        return greeting_response()

    if intent == "help":
        return help_response()

    if intent == "order_count":
        return order_count_response()

    if intent == "overall_risk":
        return overall_risk_response()

    if intent == "shipping_mode":
        return shipping_mode_response(
            message.lower()
        )

    if intent == "region":
        return region_response(
            message.lower()
        )

    if intent == "market":
        return market_response(
            message.lower()
        )

    if intent == "category":
        return category_response(
            message.lower()
        )

    if intent == "shipping_days":
        return shipping_days_response()

    if intent == "why_late":
        return why_late_response()

    if intent == "product":
        return product_response(message)

    return (
        "I'm your Supply Chain AI Assistant. "
        "I can analyze late-delivery risk, shipping modes, "
        "regions, markets, categories, products, "
        "shipping duration, and order statistics.\n\n"
        "Try asking: \"Which shipping mode has the highest risk?\""
    )