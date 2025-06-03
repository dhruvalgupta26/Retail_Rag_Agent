import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Shoe categories with realistic data for Indian shoe stores
shoes_data = {
    'Formal Shoes': {
        'items': ['Oxford Shoes', 'Derby Shoes', 'Loafers', 'Monk Strap', 'Brogues', 'Dress Boots', 'Penny Loafers'],
        'brands': ['Bata', 'Liberty', 'Red Tape', 'Woodland', 'Hush Puppies', 'Clarks', 'Lee Cooper', 'Van Heusen', 'Arrow', 'Allen Solly'],
        'colors': ['Black', 'Brown', 'Tan', 'Dark Brown', 'Cognac', 'Mahogany', 'Cherry'],
        'sizes': ['6', '7', '8', '9', '10', '11', '12', '5', '4', '13'],
        'mrp_range': (1500, 8000),
        'hsn': '6403',
        'gst_rate': 18
    },
    'Casual Shoes': {
        'items': ['Sneakers', 'Canvas Shoes', 'Boat Shoes', 'Slip-ons', 'Espadrilles', 'Deck Shoes', 'Driving Shoes'],
        'brands': ['Bata', 'Liberty', 'Sparx', 'Asian', 'Campus', 'Red Tape', 'Woodland', 'Lee Cooper', 'Fila', 'Puma'],
        'colors': ['White', 'Black', 'Navy', 'Grey', 'Blue', 'Brown', 'Beige', 'Olive'],
        'sizes': ['6', '7', '8', '9', '10', '11', '12', '5', '4', '13'],
        'mrp_range': (800, 4500),
        'hsn': '6403',
        'gst_rate': 18
    },
    'Sports Shoes': {
        'items': ['Running Shoes', 'Training Shoes', 'Basketball Shoes', 'Tennis Shoes', 'Cricket Shoes', 'Football Shoes', 'Badminton Shoes'],
        'brands': ['Nike', 'Adidas', 'Puma', 'Reebok', 'New Balance', 'Asics', 'Sparx', 'Campus', 'Nivia', 'Vector X'],
        'colors': ['White', 'Black', 'Red', 'Blue', 'Green', 'Orange', 'Yellow', 'Multi-color'],
        'sizes': ['6', '7', '8', '9', '10', '11', '12', '5', '4', '13'],
        'mrp_range': (1200, 12000),
        'hsn': '6403',
        'gst_rate': 18
    },
    'Sandals': {
        'items': ['Flip Flops', 'Sliders', 'Gladiator Sandals', 'Sports Sandals', 'Comfort Sandals', 'Beach Sandals'],
        'brands': ['Bata', 'Liberty', 'Paragon', 'Relaxo', 'Crocs', 'Adidas', 'Nike', 'Puma', 'Woodland'],
        'colors': ['Black', 'Brown', 'Blue', 'Grey', 'Green', 'Orange', 'Red'],
        'sizes': ['6', '7', '8', '9', '10', '11', '12', '5', '4'],
        'mrp_range': (300, 2500),
        'hsn': '6404',
        'gst_rate': 18
    },
    'Boots': {
        'items': ['Chelsea Boots', 'Combat Boots', 'Hiking Boots', 'Work Boots', 'Desert Boots', 'Chukka Boots', 'Ankle Boots'],
        'brands': ['Woodland', 'Red Tape', 'Timberland', 'Cat', 'Bata', 'Liberty', 'Lee Cooper', 'Provogue'],
        'colors': ['Black', 'Brown', 'Tan', 'Dark Brown', 'Olive', 'Grey'],
        'sizes': ['6', '7', '8', '9', '10', '11', '12', '5', '13'],
        'mrp_range': (2000, 15000),
        'hsn': '6403',
        'gst_rate': 18
    },
    'Ladies Footwear': {
        'items': ['Heels', 'Flats', 'Pumps', 'Wedges', 'Stilettos', 'Block Heels', 'Kitten Heels', 'Ballet Flats'],
        'brands': ['Bata', 'Liberty', 'Inc.5', 'Metro', 'Catwalk', 'Mochi', 'Carlton London', 'Lavie'],
        'colors': ['Black', 'Brown', 'Red', 'Nude', 'White', 'Pink', 'Blue', 'Gold', 'Silver'],
        'sizes': ['3', '4', '5', '6', '7', '8', '9', '10'],
        'mrp_range': (800, 6000),
        'hsn': '6403',
        'gst_rate': 18
    },
    'Kids Footwear': {
        'items': ['School Shoes', 'Sneakers', 'Sandals', 'Sports Shoes', 'Canvas Shoes', 'Slip-ons'],
        'brands': ['Bata', 'Liberty', 'Paragon', 'Action', 'Campus', 'Asian', 'Sparx'],
        'colors': ['Black', 'White', 'Blue', 'Red', 'Pink', 'Multi-color'],
        'sizes': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
        'mrp_range': (400, 2500),
        'hsn': '6403',
        'gst_rate': 18
    },
    'Ethnic Footwear': {
        'items': ['Mojaris', 'Juttis', 'Kolhapuris', 'Ethnic Sandals', 'Traditional Slippers'],
        'brands': ['Bata', 'Liberty', 'Metro', 'Mochi', 'Jaipur Footwear', 'Rajasthani Footwear'],
        'colors': ['Brown', 'Black', 'Gold', 'Silver', 'Red', 'Multi-color', 'Embroidered'],
        'sizes': ['6', '7', '8', '9', '10', '11', '5', '4'],
        'mrp_range': (500, 3500),
        'hsn': '6403',
        'gst_rate': 18
    }
}

# Indian shoe store chains and local stores
store_codes = ['SH001', 'SH002', 'SH003', 'SH004', 'SH005', 'SH006', 'SH007', 'SH008', 'SH009', 'SH010', 
               'SH011', 'SH012', 'SH013', 'SH014', 'SH015']

receipt_types = ['SALE', 'RETURN', 'EXCHANGE']

# Indian festivals and sale seasons for realistic pricing patterns
indian_festivals = [
    ('2022-04-14', '2022-04-16'),  # Baisakhi
    ('2022-08-15', '2022-08-15'),  # Independence Day
    ('2022-08-30', '2022-09-08'),  # Ganesh Chaturthi
    ('2022-10-02', '2022-10-26'),  # Dussehra to Diwali
    ('2022-11-14', '2022-11-14'),  # Diwali
    ('2022-12-25', '2023-01-01'),  # Christmas to New Year
    ('2023-03-08', '2023-03-08'),  # Holi
    ('2023-04-14', '2023-04-16'),  # Baisakhi
    ('2023-08-15', '2023-08-15'),  # Independence Day
    ('2023-08-30', '2023-09-08'),  # Ganesh Chaturthi
    ('2023-10-15', '2023-11-05'),  # Dussehra to Diwali
    ('2023-11-12', '2023-11-12'),  # Diwali
    ('2023-12-25', '2024-01-01'),  # Christmas to New Year
    ('2024-03-25', '2024-03-25'),  # Holi
    ('2024-04-14', '2024-04-16'),  # Baisakhi
    ('2024-08-15', '2024-08-15'),  # Independence Day
    ('2024-09-07', '2024-09-17'),  # Ganesh Chaturthi
    ('2024-10-12', '2024-11-01'),  # Dussehra to Diwali
    ('2024-11-01', '2024-11-01'),  # Diwali
    ('2024-12-25', '2024-12-31'),  # Christmas to New Year
]

def is_festival_season(date_obj):
    """Check if date falls in festival season for higher discounts"""
    date_str = date_obj.strftime('%Y-%m-%d')
    for start, end in indian_festivals:
        if start <= date_str <= end:
            return True
    return False

def generate_ean():
    """Generate a realistic 13-digit EAN code"""
    return ''.join([str(random.randint(0, 9)) for _ in range(13)])

def generate_pos_item_id():
    """Generate POS Item ID"""
    return f"SH{random.randint(100000, 999999)}"

def generate_shoe_invoice_data():
    """Generate realistic shoe store invoice data with multiple items per invoice"""
    
    invoice_data = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    invoice_counter = 1
    record_count = 0
    target_records = 100000
    
    while record_count < target_records:
        # Generate invoice number
        invoice_no = f"SH{str(invoice_counter).zfill(8)}"
        
        # Random store
        store_code = random.choice(store_codes)
        
        # Random date (weighted toward recent dates)
        date_weight = random.random()
        if date_weight < 0.4:  # 40% recent data (2024)
            random_start = datetime(2024, 1, 1)
            random_end = datetime(2024, 12, 31)
        elif date_weight < 0.7:  # 30% 2023 data
            random_start = datetime(2023, 1, 1)
            random_end = datetime(2023, 12, 31)
        else:  # 30% 2022 data
            random_start = datetime(2022, 1, 1)
            random_end = datetime(2022, 12, 31)
            
        random_days = random.randint(0, (random_end - random_start).days)
        invoice_date = random_start + timedelta(days=random_days)
        date_str = invoice_date.strftime('%Y-%m-%d')
        
        # Receipt type (mostly SALE)
        receipt_type = np.random.choice(receipt_types, p=[0.88, 0.08, 0.04])
        
        # Number of items in this invoice (1-6 items per invoice, shoes typically fewer items)
        num_items = np.random.choice(range(1, 7), p=[0.5, 0.25, 0.15, 0.06, 0.03, 0.01])
        
        for item_idx in range(num_items):
            if record_count >= target_records:
                break
                
            # Select random shoe category
            category = random.choice(list(shoes_data.keys()))
            cat_data = shoes_data[category]
            
            # Shoe details
            article_name = random.choice(cat_data['items'])
            brand = random.choice(cat_data['brands'])
            color = random.choice(cat_data['colors'])
            size = random.choice(cat_data['sizes'])
            
            product_desc = f"{brand} {article_name} - {color}"
            
            # Pricing
            mrp = random.randint(cat_data['mrp_range'][0], cat_data['mrp_range'][1])
            quantity = np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03])  # Shoes typically bought in pairs
            
            # Discount based on festival season and category
            if is_festival_season(invoice_date):
                # Higher discounts during festivals
                discount_pct = np.random.choice([10, 15, 20, 25, 30, 40, 50], 
                                             p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05])
            else:
                # Regular discounts
                discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], 
                                             p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
            
            discount_amount = round((mrp * quantity * discount_pct) / 100, 2)
            amount = mrp * quantity
            net_amount = round(amount - discount_amount, 2)
            
            # Indian GST calculation
            gst_rate = cat_data['gst_rate']
            gst_amount = round(net_amount * gst_rate / 100, 2)
            gross_amount = round(net_amount + gst_amount, 2)
            
            # Create record
            record = {
                'InvoiceNo': invoice_no,
                'StoreCode': store_code,
                'Date': date_str,
                'ReceiptType': receipt_type,
                'Product_Desc': product_desc,
                'EAN': generate_ean(),
                'POSItemID': generate_pos_item_id(),
                'Article_Name': article_name,
                'Colour': color,
                'Size': size,
                'Brand': brand,
                'P_Group': category,
                'HSN': cat_data['hsn'],
                'MRP': mrp,
                'Quantity': quantity,
                'Discount': discount_amount,
                'Amount': amount,
                'NetAmount': net_amount,
                'GrossAmount': gross_amount
            }
            
            invoice_data.append(record)
            record_count += 1
            
            if record_count % 10000 == 0:
                print(f"Generated {record_count} records...")
        
        invoice_counter += 1
    
    return invoice_data

# Generate the dataset
print("Generating 100,000 shoe store invoice records...")
data = generate_shoe_invoice_data()

# Create DataFrame
df = pd.DataFrame(data)

# Display summary statistics
print(f"\nDataset Summary:")
print(f"Total Records: {len(df)}")
print(f"Unique Invoices: {df['InvoiceNo'].nunique()}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Stores: {df['StoreCode'].nunique()}")
print(f"Shoe Categories: {df['P_Group'].nunique()}")

print(f"\nYear-wise Distribution:")
df['Year'] = pd.to_datetime(df['Date']).dt.year
print(df['Year'].value_counts().sort_index())

print(f"\nInvoice Distribution:")
invoice_counts = df.groupby('InvoiceNo').size()
print(f"Items per invoice - Min: {invoice_counts.min()}, Max: {invoice_counts.max()}, Avg: {invoice_counts.mean():.2f}")

print(f"\nShoe Category Distribution:")
print(df['P_Group'].value_counts())

print(f"\nTop Brands:")
print(df['Brand'].value_counts().head(10))

print(f"\nSize Distribution:")
print(df['Size'].value_counts().sort_index())

print(f"\nSample Data:")
print(df.head(10).to_string())

# Save to CSV
output_file = "shoe_store_dataset.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset saved to: {output_file}")

# Show some invoice examples
print(f"\nSample invoices with multiple items:")
sample_invoices = df['InvoiceNo'].unique()[:3]
for inv in sample_invoices:
    print(f"\n{inv}:")
    inv_data = df[df['InvoiceNo'] == inv][['Article_Name', 'Brand', 'Size', 'Quantity', 'MRP', 'NetAmount']]
    print(inv_data.to_string(index=False))

# Financial summary
print(f"\nFinancial Summary:")
print(f"Total Revenue: ₹{df['GrossAmount'].sum():,.2f}")
print(f"Average Transaction Value: ₹{df.groupby('InvoiceNo')['GrossAmount'].sum().mean():,.2f}")
print(f"Average Item Price: ₹{df['MRP'].mean():,.2f}")
print(f"Average Discount %: {(df['Discount'].sum() / df['Amount'].sum() * 100):.1f}%")