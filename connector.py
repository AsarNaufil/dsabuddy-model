import pymongo
import pandas as pd
from dotenv import load_dotenv
import os


class MongoDBConnector:
    def __init__(self, default_collection="hackrank1"):
        load_dotenv()
        self.mongo_username = os.getenv("MONGODB_ATLAS_USERNAME")
        self.mongo_password = os.getenv("MONGODB_ATLAS_PASSWORD")
        self.connection_string = f"mongodb+srv://{self.mongo_username}:{self.mongo_password}@clusterhack.gwtbxxu.mongodb.net/"
        self.client = pymongo.MongoClient(self.connection_string)
        self.default_collection = default_collection

    def get_user_data(self):
        db = self.client["problemdb"]
        collection = db["userSolvedProb"]
        cursor = collection.find({"solved_problems": {"$exists": True}})
        df = pd.DataFrame(list(cursor))
        return df

    # def get_training_problem_data(self):
    #     db = self.client["problemdb"]
    #     collection = db["hackrank1"]
    #     cursor = collection.find({"user_id": "u123456"})
    #     df = pd.DataFrame(list(cursor))
    #     return df

    def get_training_user_data(self, user_id):
        db = self.client["problemdb"]
        collection = db["userSolvedProb"]
        cursor = collection.find({"user_id": user_id})
        df = pd.DataFrame(list(cursor))
        return df

    def get_problem_data(self):
        db = self.client["problemdb"]
        collection = db["hackrank1"]
        cursor = collection.find()
        df = pd.DataFrame(list(cursor))
        return df

    def search_problems(self, criteria):
        db = self.client["problemdb"]
        collection = db["leetcode"]
        cursor = collection.find(criteria)
        return cursor
        # convert cursor to json

        # convert pd.datetime to ISODate JSON
        # problem["solved_at"] = pd.to_datetime(solved_at).to_pydatetime()

    def search_aggregate(self, pipeline):
        db = self.client["problemdb"]
        collection = db["striver"]
        cursor = collection.aggregate(pipeline)
        return cursor


# Example usage:
if __name__ == "__main__":
    connector = MongoDBConnector()
    df = connector.load_data()
    # print column names
    print(df)
