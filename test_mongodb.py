from pymongo import MongoClient
import sys
import time

def test_mongodb_connection():
    """Test MongoDB connection and basic operations"""
    print("\n=== MongoDB Connection Test ===\n")
    
    # Connection parameters
    uri = "mongodb://localhost:27017/"
    db_name = "face_verification"
    
    try:
        # Step 1: Try to connect
        print("1. Attempting to connect to MongoDB...")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)  # 5 second timeout
        
        # Force a command to test the connection
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB server")
        
        # Step 2: Access the database
        print("\n2. Accessing database:", db_name)
        db = client[db_name]
        print("✅ Successfully accessed the database")
        
        # Step 3: List all collections
        print("\n3. Existing collections:")
        collections = db.list_collection_names()
        if collections:
            for collection in collections:
                count = db[collection].count_documents({})
                print(f"   - {collection} ({count} documents)")
        else:
            print("   No collections found")
        
        # Step 4: Test write operation
        print("\n4. Testing write operation...")
        test_collection = db.test_collection
        result = test_collection.insert_one({"test": "data", "timestamp": time.time()})
        print("✅ Successfully wrote test document with ID:", result.inserted_id)
        
        # Step 5: Test read operation
        print("\n5. Testing read operation...")
        doc = test_collection.find_one({"_id": result.inserted_id})
        print("✅ Successfully read test document:", doc)
        
        # Step 6: Test delete operation
        print("\n6. Testing delete operation...")
        result = test_collection.delete_one({"_id": result.inserted_id})
        print("✅ Successfully deleted test document")
        
        # Step 7: Check database stats
        print("\n7. Database statistics:")
        stats = db.command("dbstats")
        print(f"   - Database size: {stats['dataSize']} bytes")
        print(f"   - Number of collections: {stats['collections']}")
        print(f"   - Number of objects: {stats['objects']}")
        
        print("\n✅ All MongoDB tests passed successfully!")
        return True
        
    except Exception as e:
        print("\n❌ MongoDB Connection Error:", str(e))
        print("\nTroubleshooting steps:")
        print("1. Make sure MongoDB is installed and running:")
        print("   - On macOS: brew services list")
        print("   - On Linux: sudo systemctl status mongodb")
        print("   - On Windows: Check Services app for MongoDB service")
        print("\n2. Check if MongoDB is running on default port (27017):")
        print("   - Run: lsof -i :27017")
        print("\n3. Verify MongoDB installation:")
        print("   - Run: mongosh --version")
        print("\n4. Try connecting with mongosh:")
        print("   - Run: mongosh")
        return False
    finally:
        if 'client' in locals():
            client.close()
            print("\nMongoDB connection closed")

if __name__ == "__main__":
    success = test_mongodb_connection()
    sys.exit(0 if success else 1) 