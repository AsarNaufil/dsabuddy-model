import pymongo
import pandas as pd
from dotenv import load_dotenv
import os


class MongoDBConnector:
    def __init__(self, default_collection="hackrank1"):
        load_dotenv()
        self.mongo_username = os.getenv("MONGODB_ATLAS_USERNAME")
        self.mongo_password = os.getenv("MONGODB_ATLAS_PASSWORD")
        self.connection_string = f"mongodb+srv://{self.mongo_username}:{
            self.mongo_password}@clusterhack.gwtbxxu.mongodb.net/"
        self.client = pymongo.MongoClient(self.connection_string)
        self.default_collection = default_collection

    # def load_data(self, collection_name=None):
    #     if collection_name is None:
    #         collection_name = self.default_collection
    #     db = self.client["problemdb"]
    #     collection = db[collection_name]
    #     cursor = collection.find()
    #     df = pd.DataFrame(list(cursor))
    #     return df

    def get_user_data(self):
        db = self.client["problemdb"]
        # collection = db["userData"]
        collection = db["userSolvedProb"]
        cursor = collection.find({"solved_problems": {"$exists": True}})
        df = pd.DataFrame(list(cursor))
        return df

    def get_problem_data(self):
        db = self.client["problemdb"]
        collection = db["hackrank1"]
        cursor = collection.find()
        df = pd.DataFrame(list(cursor))
        return df


# Example usage:
if __name__ == "__main__":
    connector = MongoDBConnector()
    df = connector.load_data()
    # print column names
    print(df)
