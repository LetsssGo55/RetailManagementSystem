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
