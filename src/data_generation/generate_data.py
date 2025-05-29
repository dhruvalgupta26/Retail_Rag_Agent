import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import string
import os
from tqdm import tqdm

# Initialize Faker
fake = Faker('en_IN')  # Using Indian locale for realistic Indian retail data
np.random.seed(42)
random.seed(42)

class RetailDataGenerator:
    def __init__(self):
        # Brand and product data
        self.brands = ['Nike', 'Adidas', 'Puma', 'Reebok', 'Fila', 'Skechers', 'Bata', 
                      'Woodland', 'Liberty', 'Red Tape', 'Sparx', 'Campus', 'Asian', 'VKC']
        
        self.colors = ['Black', 'White', 'Brown', 'Tan', 'Navy', 'Grey', 'Red', 'Blue', 
                      'Green', 'Yellow', 'Pink', 'Purple', 'Orange', 'Maroon', 'Beige']
        
        self.sizes = ['6', '7', '8', '9', '10', '11', '12', '13', '5', '4', 'XS', 'S', 'M', 'L', 'XL', 'XXL']
        
        self.product_types = ['Sneakers', 'Formal Shoes', 'Sandals', 'Boots', 'Loafers', 
                             'Sports Shoes', 'Casual Shoes', 'Flip Flops', 'Heels', 'Flats']
        
        self.leather_types = ['Genuine Leather', 'Synthetic', 'Patent Leather', 'Suede', 
                             'Canvas', 'Mesh', 'Rubber', 'Fabric']
        
        self.sole_materials = ['Rubber', 'EVA', 'PU', 'TPR', 'Leather', 'Synthetic']
        
        self.seasons = ['Spring', 'Summer', 'Monsoon', 'Winter', 'All Season']
        
        # Indian cities for stores
        self.cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
                      'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 
                      'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara']
        
        self.states = {'Mumbai': 'Maharashtra', 'Delhi': 'Delhi', 'Bangalore': 'Karnataka',
                      'Chennai': 'Tamil Nadu', 'Kolkata': 'West Bengal', 'Hyderabad': 'Telangana',
                      'Pune': 'Maharashtra', 'Ahmedabad': 'Gujarat', 'Jaipur': 'Rajasthan',
                      'Lucknow': 'Uttar Pradesh', 'Kanpur': 'Uttar Pradesh', 'Nagpur': 'Maharashtra',
                      'Indore': 'Madhya Pradesh', 'Thane': 'Maharashtra', 'Bhopal': 'Madhya Pradesh',
                      'Visakhapatnam': 'Andhra Pradesh', 'Patna': 'Bihar', 'Vadodara': 'Gujarat'}
        
        # Initialize reference data
        self.stores_df = None
        self.dcs_df = None
        self.items_df = None
        
    def generate_dcs_data(self, n_departments=10):
        """Generate DCS (Department-Class-Subclass) hierarchy"""
        print("Generating DCS hierarchy...")
        
        departments = [
            ('FOOTWEAR', ['Men Footwear', 'Women Footwear', 'Kids Footwear', 'Sports Footwear']),
            ('ACCESSORIES', ['Belts', 'Wallets', 'Bags', 'Socks']),
            ('APPAREL', ['Shirts', 'T-Shirts', 'Jeans', 'Trousers']),
        ]
        
        dcs_data = []
        dept_id = 1
        class_id = 1
        subclass_id = 1
        
        for dept_name, classes in departments:
            for class_name in classes:
                # Generate subclasses for each class
                subclasses = [f"{class_name} - Premium", f"{class_name} - Regular", 
                             f"{class_name} - Economy", f"{class_name} - Sports"]
                
                for subclass_name in subclasses:
                    dcs_data.append({
                        'DepartmentID': dept_id,
                        'DepartmentName': dept_name,
                        'ClassID': class_id,
                        'ClassName': class_name,
                        'SubClassID': subclass_id,
                        'SubClassName': subclass_name
                    })
                    subclass_id += 1
                class_id += 1
            dept_id += 1
        
        self.dcs_df = pd.DataFrame(dcs_data)
        return self.dcs_df
    
    def generate_stores_data(self, n_stores=500):
        """Generate store master data"""
        print(f"Generating {n_stores} stores...")
        
        stores_data = []
        for i in tqdm(range(n_stores), desc="Creating stores"):
            city = random.choice(self.cities)
            state = self.states[city]
            
            store_data = {
                'ID': i + 1,
                'Code': f"ST{i+1:04d}",
                'BName': f"{fake.company()} - {city}",
                'Address': fake.address(),
                'City': city,
                'State': state,
                'PostalCode': fake.postcode(),
                'Phone': fake.phone_number(),
                'sapStoreCode': f"SAP{i+1:04d}",
                'sapStoreTypeCode': random.choice(['RETAIL', 'FLAGSHIP', 'OUTLET', 'FRANCHISE']),
                'storeZone': random.choice(['North', 'South', 'East', 'West', 'Central']),
                'storeLocation': random.choice(['Mall', 'Street', 'Shopping Complex', 'Standalone']),
                'storeCarpetArea': random.randint(500, 3000),
                'areaGrading': random.choice(['A+', 'A', 'B+', 'B', 'C']),
                'rsmName': fake.name(),
                'rsmEmail': fake.email(),
                'rsmEmployeeCode': f"RSM{random.randint(1000, 9999)}",
                'asmName': fake.name(),
                'asmEmail': fake.email(),
                'asmEmployeeCode': f"ASM{random.randint(1000, 9999)}",
                'serverID': f"SRV{random.randint(100, 999)}",
                'GSTIN': f"{random.randint(10, 37)}{''.join([random.choice(string.ascii_uppercase) for _ in range(5)])}{random.randint(1000, 9999)}{random.choice(string.ascii_uppercase)}{random.randint(1, 9)}Z{random.randint(1, 9)}",
                'businessPhone': fake.phone_number(),
                'landline': fake.phone_number(),
                'removedDetail': None if random.random() > 0.1 else fake.date(),
                'merchandiser': fake.name(),
                'merchandiser1': fake.name() if random.random() > 0.3 else None,
                'customRegion': f"REG{random.randint(1, 10)}",
                'StoreCode': f"ST{i+1:04d}",
                'storeManager': fake.name(),
                'lic': f"LIC{random.randint(10000, 99999)}",
                'emailID': fake.email()
            }
            stores_data.append(store_data)
        
        self.stores_df = pd.DataFrame(stores_data)
        return self.stores_df
    
    def generate_items_data(self, n_items=50000):
        """Generate item master data"""
        print(f"Generating {n_items} items...")
        
        items_data = []
        
        for i in tqdm(range(n_items), desc="Creating items"):
            brand = random.choice(self.brands)
            color = random.choice(self.colors)
            size = random.choice(self.sizes)
            product_type = random.choice(self.product_types)
            leather_type = random.choice(self.leather_types)
            sole_material = random.choice(self.sole_materials)
            season = random.choice(self.seasons)
            
            # Generate realistic costs and MRP
            cost = random.randint(200, 3000)
            mrp = cost * random.uniform(1.5, 3.0)
            
            # Select random DCS entry
            dcs_entry = self.dcs_df.sample(1).iloc[0]
            
            item_data = {
                'ReportingMHLevel1': dcs_entry['DepartmentName'],
                'DCS_CODE': f"{dcs_entry['DepartmentID']:02d}{dcs_entry['ClassID']:02d}{dcs_entry['SubClassID']:02d}",
                'VEND_CODE': f"V{random.randint(1000, 9999)}",
                'PRODUCT_DESC': f"{brand} {product_type} {color}",
                'ARTICLE_NAME': f"{brand} {product_type}",
                'COLOUR': color,
                'Size': size,
                'BRAND': brand,
                'EAN': f"{random.randint(100000000000, 999999999999)}",
                'BARCODE': f"BC{random.randint(1000000, 9999999)}",
                'PRC_AT': random.choice(['Regular', 'Premium', 'Sale']),
                'HEEL_HEIGHT': random.choice([0, 1, 2, 3, 4, 5, 6]) if 'Heel' in product_type else 0,
                'SOLE_MATERIAL': sole_material,
                'SEASON': season,
                'COST': round(cost, 2),
                'ORDCOST': round(cost * random.uniform(0.95, 1.05), 2),
                'MRP': round(mrp, 2),
                'ItemDEPT': dcs_entry['DepartmentName'],
                'CLASS': dcs_entry['ClassName'],
                'SUBCLASS': dcs_entry['SubClassName'],
                'SOLE': sole_material,
                'FW_TYPE': product_type,
                'LETHR_CATEGORY': random.choice(['Premium', 'Regular', 'Economy']),
                'LETHR_TYPE': leather_type,
                'ITEM_FLAG': random.choice(['A', 'I', 'D']),  # Active, Inactive, Discontinued
                'P_GROUP': f"PG{random.randint(100, 999)}",
                'INDICATOR': random.choice(['NEW', 'REG', 'SALE', 'DISC'])
            }
            items_data.append(item_data)
        
        self.items_df = pd.DataFrame(items_data)
        return self.items_df
    
    def generate_stock_data(self, n_records=200000):
        """Generate stock data"""
        print(f"Generating {n_records} stock records...")
        
        stock_data = []
        
        for i in tqdm(range(n_records), desc="Creating stock records"):
            # Select random store and item
            store = self.stores_df.sample(1).iloc[0]
            item = self.items_df.sample(1).iloc[0]
            
            # Generate realistic stock quantities
            current_stock = random.randint(0, 100)
            fresh_defective = random.randint(0, min(5, current_stock))
            shop_soil = random.randint(0, min(3, current_stock))
            cust_claim = random.randint(0, min(2, current_stock))
            
            # Calculate pending RTV (Return to Vendor)
            pending_rtv_fresh = random.randint(0, fresh_defective)
            pending_rtv_shop = random.randint(0, shop_soil)
            
            total_stock = current_stock + fresh_defective + shop_soil + cust_claim
            
            # Calculate values based on MRP
            mrp = float(item['MRP'])
            current_stock_value = current_stock * mrp
            fresh_defective_value = fresh_defective * mrp * 0.5  # Defective at 50% value
            shop_soil_value = shop_soil * mrp * 0.7  # Shop soil at 70% value
            cust_claim_value = cust_claim * mrp
            
            stock_record = {
                'STORE': store['Code'],
                'SKU': f"SKU{i+1:08d}",
                'EAN': item['EAN'],
                'MATERIAL_BARCODE': item['BARCODE'],
                'COLOR': item['COLOUR'],
                'SIZE': item['Size'],
                'ARTICLE': f"ART{random.randint(10000, 99999)}",
                'BRAND NAME': item['BRAND'],
                'P GROUP': item['P_GROUP'],
                'MRP': mrp,
                'CURRENT_STOCK': current_stock,
                'CURRENT_STOCK_VALUE': round(current_stock_value, 2),
                'FRESH_DEFECTIVE_STOCK': fresh_defective,
                'FRESH_DEFECTIVE_STOCK_VALUE': round(fresh_defective_value, 2),
                'PENDING_RTV_FRESH_DEFECTIVE_STOCK': pending_rtv_fresh,
                'PENDING_RTV_FRESH_DEFECTIVE_STOCK_VALUE': round(pending_rtv_fresh * mrp * 0.5, 2),
                'SHOP_SOIL_STOCK': shop_soil,
                'SHOP_SOIL_STOCK_VALUE': round(shop_soil_value, 2),
                'PENDING_RTV_SHOP_SOIL_STOCK': pending_rtv_shop,
                'PENDING_RTV_SHOP_SOIL_STOCK_VALUE': round(pending_rtv_shop * mrp * 0.7, 2),
                'CUST_CLAIM_STOCK': cust_claim,
                'CUST_CLAIM_STOCK_VALUE': round(cust_claim_value, 2),
                'TOTAL_STOCK': total_stock,
                'TOTAL_VALUE': round(current_stock_value + fresh_defective_value + shop_soil_value + cust_claim_value, 2)
            }
            stock_data.append(stock_record)
        
        return pd.DataFrame(stock_data)
    
    def generate_sales_data(self, n_records=700000):
        """Generate sales transaction data"""
        print(f"Generating {n_records} sales records...")
        
        sales_data = []
        
        # Date range for sales (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        for i in tqdm(range(n_records), desc="Creating sales records"):
            # Select random store and item
            store = self.stores_df.sample(1).iloc[0]
            item = self.items_df.sample(1).iloc[0]
            
            # Generate transaction details
            transaction_date = fake.date_between(start_date=start_date, end_date=end_date)
            quantity = random.randint(1, 5)
            mrp = float(item['MRP'])
            
            # Generate discount (0-30%)
            discount_percent = random.uniform(0, 0.3)
            discount_amount = round(mrp * quantity * discount_percent, 2)
            net_amount = round(mrp * quantity - discount_amount, 2)
            
            # Add GST (18% for footwear)
            gst_amount = round(net_amount * 0.18, 2)
            gross_amount = round(net_amount + gst_amount, 2)
            
            sales_record = {
                'InvoiceNo': f"INV{random.randint(100000000, 999999999)}",
                'StoreCode': store['Code'],
                'Date': transaction_date.strftime('%Y-%m-%d'),
                'ReceiptType': random.choice(['SALE', 'RETURN', 'EXCHANGE']) if random.random() > 0.85 else 'SALE',
                'Product_Desc': item['PRODUCT_DESC'],
                'EAN': item['EAN'],
                'POSItemID': f"POS{random.randint(100000, 999999)}",
                'Article_Name': item['ARTICLE_NAME'],
                'Colour': item['COLOUR'],
                'Size': item['Size'],
                'Brand': item['BRAND'],
                'P_Group': item['P_GROUP'],
                'HSN': '6403',  # HSN code for footwear
                'MRP': mrp,
                'Quantity': quantity,
                'DiscountAmount': discount_amount,
                'NetAmount': net_amount,
                'GrossAmount': gross_amount
            }
            sales_data.append(sales_record)
        
        return pd.DataFrame(sales_data)
    
    def generate_all_data(self):
        """Generate all datasets"""
        print("Starting synthetic data generation...")
        print("=" * 50)
        
        # Generate reference data first
        dcs_df = self.generate_dcs_data()
        stores_df = self.generate_stores_data(500)
        items_df = self.generate_items_data(50000)
        
        # Generate transactional data
        stock_df = self.generate_stock_data(200000)
        sales_df = self.generate_sales_data(250000)  # Increased to reach 1M total
        
        # Summary
        total_records = len(dcs_df) + len(stores_df) + len(items_df) + len(stock_df) + len(sales_df)
        
        print("\n" + "=" * 50)
        print("GENERATION COMPLETE!")
        print("=" * 50)
        print(f"DCS Records: {len(dcs_df):,}")
        print(f"Store Records: {len(stores_df):,}")
        print(f"Item Records: {len(items_df):,}")
        print(f"Stock Records: {len(stock_df):,}")
        print(f"Sales Records: {len(sales_df):,}")
        print(f"TOTAL RECORDS: {total_records:,}")
        print("=" * 50)
        
        return {
            'DCS_Header': dcs_df,
            'Store_Header': stores_df,
            'Item_Header': items_df,
            'Stock_Header': stock_df,
            'Sale_Header': sales_df
        }
    
    def save_to_excel(self, data_dict, filename='retail_synthetic_data.xlsx'):
        """Save all data to Excel file"""
        print(f"\nSaving data to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                print(f"Writing {sheet_name}: {len(df):,} records")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Data saved successfully to {filename}")
        
        # Print file size
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"📁 File size: {file_size:.2f} MB")
    
    def save_to_csv(self, data_dict, folder='retail_data_csv'):
        """Save all data to separate CSV files"""
        print(f"\nSaving data to CSV files in {folder}/...")
        
        os.makedirs(folder, exist_ok=True)
        
        for sheet_name, df in data_dict.items():
            filename = f"{folder}/{sheet_name}.csv"
            print(f"Writing {filename}: {len(df):,} records")
            df.to_csv(filename, index=False)
        
        print(f"✅ All CSV files saved in {folder}/ directory")

def main():
    """Main execution function"""
    print("🏪 RETAIL SYNTHETIC DATA GENERATOR")
    print("🎯 Target: 1,000,000+ Records")
    print("📊 Production-Grade Dataset")
    print("=" * 50)
    
    # Initialize generator
    generator = RetailDataGenerator()
    
    # Generate all data
    all_data = generator.generate_all_data()
    
    # Save data
    print("\n📁 SAVING DATA...")
    generator.save_to_excel(all_data)
    generator.save_to_csv(all_data)
    
    print("\n🎉 SYNTHETIC DATA GENERATION COMPLETED!")
    print("✨ Ready for testing and analysis")

if __name__ == "__main__":
    # Install required packages
    required_packages = ['pandas', 'numpy', 'faker', 'tqdm', 'openpyxl']
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Please install {package}: pip install {package}")
    
    main()