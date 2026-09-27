def air_transport_system(order):
    return {
        "mode": "AIR",
        "icon": "✈️",
        "estimated_days": 2,
        "cost_level": "HIGH",
        "capacity": "AVAILABLE",
        "status": "Ready for booking",
        "message": "Air transport can be assigned for fast delivery."
    }


def road_transport_system(order):
    return {
        "mode": "ROAD",
        "icon": "🚚",
        "estimated_days": 4,
        "cost_level": "MEDIUM",
        "capacity": "AVAILABLE",
        "status": "Ready for dispatch",
        "message": "Road transport is suitable for domestic shipments."
    }


def sea_transport_system(order):
    return {
        "mode": "SEA",
        "icon": "🚢",
        "estimated_days": 12,
        "cost_level": "LOW",
        "capacity": "AVAILABLE",
        "status": "Ready for scheduling",
        "message": "Sea transport is suitable for low-cost international shipping."
    }


def get_transport_options(order):

    return [
        air_transport_system(order),
        road_transport_system(order),
        sea_transport_system(order)
    ]


def logistics_orchestrator(
    delivery_risk,
    international=False,
    urgent=False,
    cost_sensitive=False
):

    order = {
        "delivery_risk": delivery_risk,
        "international": international,
        "urgent": urgent,
        "cost_sensitive": cost_sensitive
    }

    options = get_transport_options(order)

    if delivery_risk == 1 and urgent:

        selected = options[0]

        reason = (
            "High delivery risk and urgent shipment detected. "
            "AI selected AIR transport to minimize transit time."
        )

        priority = "CRITICAL"

    elif international and cost_sensitive:

        selected = options[2]

        reason = (
            "International shipment is cost-sensitive. "
            "AI selected SEA transport to reduce transportation cost."
        )

        priority = "NORMAL"

    elif international:

        selected = options[0]

        reason = (
            "International shipment requires relatively fast movement. "
            "AI selected AIR transport."
        )

        priority = "HIGH"

    else:

        selected = options[1]

        reason = (
            "Domestic shipment detected. "
            "AI selected ROAD transport."
        )

        priority = "NORMAL"

    if delivery_risk == 1:
        action = "Prioritize shipment"
    else:
        action = "Normal processing"

    return {
        "risk_status": (
            "HIGH RISK"
            if delivery_risk == 1
            else "LOW RISK"
        ),

        "selected_mode": selected["mode"],
        "transport_icon": selected["icon"],
        "estimated_days": selected["estimated_days"],
        "cost_level": selected["cost_level"],
        "capacity": selected["capacity"],
        "transport_status": selected["status"],
        "priority": priority,
        "action": action,
        "reason": reason,
        "message": selected["message"],
        "all_options": options
    }


def generate_risk_report(df):

    total_orders = len(df)

    high_risk_count = len(
        df[df["Late_delivery_risk"] == 1]
    )

    low_risk_count = len(
        df[df["Late_delivery_risk"] == 0]
    )

    percentage = (
        high_risk_count / total_orders * 100
        if total_orders > 0
        else 0
    )

    return {
        "total_orders": total_orders,
        "high_risk_orders": high_risk_count,
        "low_risk_orders": low_risk_count,
        "high_risk_percentage": round(percentage, 2),
        "recommended_action": "Prioritize high-risk orders"
    }


def identify_high_risk_orders(df):

    if "Late_delivery_risk" not in df.columns:
        return df.iloc[0:0]

    return df[
        df["Late_delivery_risk"] == 1
    ].copy()