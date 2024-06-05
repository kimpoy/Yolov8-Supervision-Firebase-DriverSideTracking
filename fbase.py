import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase with the credentials JSON file
cred = credentials.Certificate("iot-project-6b313-firebase-adminsdk-l3dnb-b8239e308b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-project-6b313-default-rtdb.firebaseio.com/'
})

# Get a reference to the Firebase Realtime Database
ref = db.reference('/Counter')

""" # Create
def create_data(data):
    new_data_ref = ref.push()
    new_data_ref.set(data) """

""" # Read
def read_data():
    return ref.get() """

# Update
def update_data(data_id, updated_data):
    ref.child(data_id).update(updated_data)

""" # Delete
def delete_data(data_id):
    ref.child(data_id).delete()
 """
# Example usage
if __name__ == "__main__":
    """ # Create data
    data = {"Counter": 0}
    create_data(data) """

    """ # Read data
    print("Data before update:")
    print(read_data()) """

    # Update data
    data_id = "-NzdK5eOSWNahdBKg3EP"  # Replace <data_id> with the ID of the data you want to update
    updated_data = {"Counter": 0}
    update_data(data_id, updated_data)

    """ # Read data after update
    print("Data after update:")
    print(read_data())
 """
    """ # Delete data
    delete_data(data_id)
    print("Data after deletion:")
    print(read_data()) """
