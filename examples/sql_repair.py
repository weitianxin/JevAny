"""Select a query repair, execute it in SQLite, and check the result."""
import json
import sqlite3

from jevany import Choice
from ._common import client, parser

ORDERS = [(1, "North", 60, "paid"), (2, "North", 60, "paid"),
          (3, "South", 35, "paid"), (4, "West", 20, "draft"), (5, "South", 90, "paid")]
ITEMS = [(1, 0), (1, 0), (2, 0), (3, 0), (3, None), (4, 0), (5, 1)]
QUERIES = {
    "join": "SELECT o.region, SUM(o.total) FROM orders o JOIN items i ON o.id=i.order_id WHERE o.status='paid' AND i.returned=0 GROUP BY o.region",
    "distinct_sum": "SELECT o.region, SUM(DISTINCT o.total) FROM orders o JOIN items i ON o.id=i.order_id WHERE o.status='paid' AND i.returned=0 GROUP BY o.region",
    "exists": "SELECT o.region, SUM(o.total) FROM orders o WHERE o.status='paid' AND EXISTS (SELECT 1 FROM items i WHERE i.order_id=o.id AND i.returned=0) GROUP BY o.region",
    "all_orders": "SELECT region, SUM(total) FROM orders WHERE status='paid' GROUP BY region",
}


def run(decide) -> dict:
    response = decide.system_one(
        state={"orders_columns": ["id", "region", "total", "status"], "orders": [list(row) for row in ORDERS],
               "items_columns": ["order_id", "returned"], "items": [list(row) for row in ITEMS],
               "current_query": QUERIES["join"]},
        questions={"repair": Choice(
            instructions="Choose the query that sums each paid order exactly once if it has a non-returned item. Distinct orders can have the same total.",
            criteria=QUERIES,
        )},
    )
    selected = response["answers"]["repair"]["choice"]
    with sqlite3.connect(":memory:") as database:
        database.execute("CREATE TABLE orders(id INTEGER, region TEXT, total INTEGER, status TEXT)")
        database.execute("CREATE TABLE items(order_id INTEGER, returned INTEGER)")
        database.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", ORDERS)
        database.executemany("INSERT INTO items VALUES (?, ?)", ITEMS)
        actual = dict(database.execute(QUERIES[selected]).fetchall())
    expected = {}
    for identifier, region, total, status in ORDERS:
        if status == "paid" and any(order == identifier and returned == 0 for order, returned in ITEMS):
            expected[region] = expected.get(region, 0) + total
    return {"decision": response, "actual": actual, "expected": expected, "passed": actual == expected}


def main():
    args = parser(__doc__).parse_args()
    result = run(client(args))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
