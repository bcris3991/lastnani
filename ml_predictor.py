"""
ml_predictor.py
---------------
Demand Prediction for WMSU Book Inventory System
Gamit: Linear Regression (scikit-learn)

Gi-predict niini ang expected borrow count sa matag book
base sa historical borrow data (per DAY).
"""

import sqlite3
from datetime import datetime

import numpy as np

# Graceful import — mag-warn lang kung wala pa gi-install ang scikit-learn
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─── MAIN PREDICTION FUNCTION ─────────────────────────────────────────────────

def predict_demand(db_path: str) -> list[dict]:
    """
    Basaha ang borrow_requests table ug i-predict ang demand sa sunod na ADLAW
    para sa matag book/item.

    Returns:
        list of dicts:
        [
            {
                'item_id': int,
                'item_name': str,
                'category': str,
                'avg_daily_borrows': float,
                'predicted_tomorrow': int,
                'trend': str,          # 'Rising', 'Stable', 'Declining'
                'confidence': str,     # 'High', 'Medium', 'Low'
                'total_borrows': int,
                'days_of_data': int,
            },
            ...
        ]
    """
    if not SKLEARN_AVAILABLE:
        return _fallback_simple_average(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Kuhaon ang tanan borrow records per DAY (Approved + Returned lang)
    rows = cur.execute("""
        SELECT
            br.item_id,
            i.item_name,
            i.category,
            strftime('%Y-%m-%d', br.created_at) AS borrow_day,
            COUNT(*) AS borrow_count
        FROM borrow_requests br
        JOIN items i ON br.item_id = i.item_id
        WHERE br.status IN ('Approved', 'Returned')
        GROUP BY br.item_id, borrow_day
        ORDER BY br.item_id, borrow_day
    """).fetchall()

    conn.close()

    if not rows:
        return _fallback_with_pending(db_path)

    # Organize data per item
    items_data: dict[int, dict] = {}
    for row in rows:
        iid = row['item_id']
        if iid not in items_data:
            items_data[iid] = {
                'item_name': row['item_name'],
                'category': row['category'],
                'daily': {}
            }
        items_data[iid]['daily'][row['borrow_day']] = row['borrow_count']

    results = []
    for item_id, data in items_data.items():
        daily = data['daily']
        days_sorted = sorted(daily.keys())
        counts = [daily[d] for d in days_sorted]
        n = len(counts)

        avg = sum(counts) / n
        total = sum(counts)

        if n >= 2:
            # Linear regression: X = day index, y = borrow count
            X = np.array(range(n)).reshape(-1, 1)
            y = np.array(counts, dtype=float)
            model = LinearRegression()
            model.fit(X, y)
            predicted = max(0, round(model.predict([[n]])[0]))
            slope = model.coef_[0]

            if slope > 0.3:
                trend = 'Rising'
            elif slope < -0.3:
                trend = 'Declining'
            else:
                trend = 'Stable'

            if n >= 14:
                confidence = 'High'
            elif n >= 7:
                confidence = 'Medium'
            else:
                confidence = 'Low'
        else:
            # Only 1 day of data
            predicted = round(avg)
            trend = 'Stable'
            confidence = 'Low'

        results.append({
            'item_id': item_id,
            'item_name': data['item_name'],
            'category': data['category'],
            'avg_daily_borrows': round(avg, 1),
            'predicted_tomorrow': predicted,
            'trend': trend,
            'confidence': confidence,
            'total_borrows': total,
            'days_of_data': n,
        })

    # Sort by predicted demand (highest first)
    results.sort(key=lambda x: x['predicted_tomorrow'], reverse=True)
    return results


def _fallback_with_pending(db_path: str) -> list[dict]:
    """
    Fallback: kung wala pay Approved/Returned records,
    gamiton ang Pending requests para makakuha ug datos.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT
            br.item_id,
            i.item_name,
            i.category,
            COUNT(*) as total_requests
        FROM borrow_requests br
        JOIN items i ON br.item_id = i.item_id
        GROUP BY br.item_id
        ORDER BY total_requests DESC
    """).fetchall()

    conn.close()

    results = []
    for row in rows:
        results.append({
            'item_id': row['item_id'],
            'item_name': row['item_name'],
            'category': row['category'],
            'avg_daily_borrows': float(row['total_requests']),
            'predicted_tomorrow': row['total_requests'],
            'trend': 'Stable',
            'confidence': 'Low',
            'total_borrows': row['total_requests'],
            'days_of_data': 1,
        })
    return results


def _fallback_simple_average(db_path: str) -> list[dict]:
    """
    Fallback kung wala gi-install si scikit-learn.
    Gamiton ang simple daily average ra.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT
            br.item_id,
            i.item_name,
            i.category,
            COUNT(*) as total,
            COUNT(DISTINCT strftime('%Y-%m-%d', br.created_at)) as days
        FROM borrow_requests br
        JOIN items i ON br.item_id = i.item_id
        WHERE br.status IN ('Approved', 'Returned', 'Pending')
        GROUP BY br.item_id
        ORDER BY total DESC
    """).fetchall()

    conn.close()

    results = []
    for row in rows:
        days = max(row['days'], 1)
        avg = row['total'] / days
        results.append({
            'item_id': row['item_id'],
            'item_name': row['item_name'],
            'category': row['category'],
            'avg_daily_borrows': round(avg, 1),
            'predicted_tomorrow': round(avg),
            'trend': 'Stable',
            'confidence': 'Low (install scikit-learn for ML)',
            'total_borrows': row['total'],
            'days_of_data': days,
        })
    return results


def get_category_summary(predictions: list[dict]) -> list[dict]:
    """
    I-group ang predictions by category para sa chart.
    """
    summary: dict[str, int] = {}
    for p in predictions:
        cat = p['category']
        summary[cat] = summary.get(cat, 0) + p['predicted_tomorrow']

    return [
        {'category': cat, 'predicted_demand': total}
        for cat, total in sorted(summary.items(), key=lambda x: x[1], reverse=True)
    ]


def predict_low_stock(db_path: str) -> list[dict]:
    """
    ML-powered low stock prediction.

    Gi-combine ang:
    1. Current quantity
    2. Predicted daily demand (from predict_demand)
    3. Borrow rate trend

    Returns items predicted to run low within the next 7 days,
    sorted by urgency (days until stockout).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Kuhaon ang tanan items with their current stock
    items = cur.execute(
        "SELECT item_id, item_name, category, quantity, status FROM items ORDER BY item_name"
    ).fetchall()
    conn.close()

    if not items:
        return []

    # Get demand predictions
    demand_preds = {p['item_id']: p for p in predict_demand(db_path)}

    results = []
    for item in items:
        iid = item['item_id']
        current_qty = item['quantity']
        pred = demand_preds.get(iid)

        if pred:
            daily_demand = max(pred['avg_daily_borrows'], 0.1)
            predicted_tomorrow = pred['predicted_tomorrow']
            trend = pred['trend']
        else:
            # No borrow history — assume low baseline demand
            daily_demand = 0.5
            predicted_tomorrow = 0
            trend = 'Stable'

        # Days until stockout
        if daily_demand > 0:
            days_until_stockout = current_qty / daily_demand
        else:
            days_until_stockout = 999

        # Urgency score — lower = more urgent
        if current_qty == 0:
            urgency = 'Critical'
            days_label = 'Out of Stock'
        elif days_until_stockout <= 3:
            urgency = 'High'
            days_label = f'~{int(days_until_stockout)} day(s)'
        elif days_until_stockout <= 7:
            urgency = 'Medium'
            days_label = f'~{int(days_until_stockout)} day(s)'
        elif days_until_stockout <= 14:
            urgency = 'Low'
            days_label = f'~{int(days_until_stockout)} day(s)'
        else:
            continue  # Skip items that are well-stocked

        # Extra risk if trend is rising
        risk_flag = trend == 'Rising' and days_until_stockout <= 14

        results.append({
            'item_id': iid,
            'item_name': item['item_name'],
            'category': item['category'],
            'current_quantity': current_qty,
            'status': item['status'],
            'avg_daily_demand': round(daily_demand, 1),
            'predicted_tomorrow': predicted_tomorrow,
            'days_until_stockout': round(days_until_stockout, 1),
            'days_label': days_label,
            'urgency': urgency,
            'trend': trend,
            'risk_flag': risk_flag,
        })

    # Sort: Critical first, then by days_until_stockout ascending
    urgency_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    results.sort(key=lambda x: (urgency_order.get(x['urgency'], 9), x['days_until_stockout']))
    return results