from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_verification"]

# Drop the users collection
db.users.drop()

print("Database cleared successfully") 