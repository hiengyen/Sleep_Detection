import firebase_admin
from firebase_admin import credentials, db


# Init  Firebase Admin SDK
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(
    cred,
    {
        "storageBuckage": "sleepdetection-43473.firebasestorage.app",
        "databaseURL": "https://sleepdetection-43473-default-rtdb.asia-southeast1.firebasedatabase.app/",
    },
)


def init_db():
    ref_users = db.reference("users")
    users = [
        {
            "user_id": "01",
            "name": "user1",
            "email": "user1@gmail.com",
            "phone": "0923899497",
        },
    ]

    if not ref_users.get():  # if branch not exists
        for user in users:
            ref_users.child(user["user_id"]).set(
                {
                    "name": user["name"],
                    "email": user["email"],
                    "phone": user["phone"],
                }
            )


def update_message(detail, message_type):
    """
    Cập nhật thông `message` trong Firebase Realtime Database.

    :param message_type: Loại trạng thái (e.g., "normal","detect" ,"undetect","unconcentrate").
    :param detail: Thông tin chi tiết liên quan đến trạng thái.
    """
    ref = db.reference("message")
    ref.update(
        {
            "detail": detail,
            "message_type": message_type,
        }
    )


init_db()
