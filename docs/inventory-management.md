# Inventory management

The web UI's **Inventory** section is the operational stock-control screen.

## Add a catalogue item

Enter a unique SKU, name, category, unit and pricing details. An optional barcode
can be registered during product creation. Opening stock is recorded as an
`OPENING_BALANCE` movement attributed to the signed-in user.

## Adjust stock manually

Choose a product and select **Add stock** or **Remove stock**, enter a positive
whole-unit quantity, and provide a reason. The adjustment and its audit-log row
are written atomically. Stock cannot be reduced below zero.

## Barcode scanner workflow

A USB/Bluetooth barcode scanner in keyboard-wedge mode can be focused on the
barcode input; each scan is resolved against registered barcodes and product
SKUs. Each scan adds one unit to the scan cart. Press **Create scanned order**
to generate a draft order. Unknown barcodes remain unresolved and must be
resolved in the order review queue before fulfilment.

## Receipt OCR workflow

Upload an image or PDF in **OCR Intake**. The existing Tesseract OCR and product
matcher create a draft order, classify each matched product against current
stock, and show availability. PDF support uses `pypdfium2` to render up to
8 pages; uploads are limited to 15 MB. Install Python dependencies with
`pip install -r requirements.txt`, and install the Tesseract executable
separately on the host.

## Confirm an order

Use **Review & confirm** in the Orders view. The app re-checks availability and
shows ordered and available quantities. After confirmation, the app deducts
only the available quantity, marks the order fulfilled, and creates an invoice.
Stock deductions are protected from going negative and a repeated fulfilment
attempt is rejected.

## Audit history

The stock movement log records the movement type, signed quantity, before/after
balances, time, signed-in user, note, and order reference when applicable.
Movement types include `OPENING_BALANCE`, `MANUAL_ADD`, `MANUAL_REMOVE`,
and `ORDER_FULFILMENT`.

Removing units is a stock adjustment; catalogue products are retained for
historic order and audit references rather than physically deleted.


## Multi-buyer receipts and consolidated picking

Before uploading a receipt or creating a barcode order, select the buyer. New
buyers can be registered in the Customers section. Receipt and scanner orders
retain that buyer ID, independently of the signed-in operator who later
fulfils the order.

The Orders page builds a read-only picking wave across every open order. It
allocates the shared stock snapshot once, prioritising fully fulfilable orders,
then higher standalone fulfilment percentage, then FIFO. Identical products
are combined into category / K-Means sales-tier / cluster baskets, with the
requested, pickable, short and remaining quantities shown along with each
buyer's share. The plan does not alter stock; actual deductions occur only
when the next-priority order is confirmed and fulfilled. The server rejects
fulfilment out of priority order, and the wave is recalculated after every
completed order.

Frequently bought-together suggestions come from the persisted FP-Growth
association rules in `bundle_recommendations`. Suggestions are advisory and
are not silently added to a buyer's order. They are hidden when there is no
stock left after planned picking.

To refresh these associations after new sales / receipt orders have accumulated,
run `python main.py train`. The training command persists the new rules for
the picking-wave suggestion panel. If there are no saved association rules,
the suggestion table remains empty rather than inventing product pairs.


## Buyer-aware stock audit file

The SQLite `inventory_movements` table remains the authoritative ledger.
Existing on-hand stock is imported once as `OPENING_BALANCE` movements when
the audit schema is first enabled. Each subsequent movement records its
timestamp, product, signed quantity, stock before and after, buyer (when tied
to an order), signed-in operator, reason and order reference.

A CSV projection is written to `logs/inventory_movements.csv` beside the
configured database file. The Inventory page's **Download CSV log** action
refreshes and downloads that complete ledger. Columns distinguish the
customer/buyer from the operator who performed the action.
