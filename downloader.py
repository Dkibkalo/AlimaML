import instaloader
import shutil
import os

def download_instagram_photos(username):
    shutil.rmtree("photos", ignore_errors=True)
    os.makedirs("photos", exist_ok=True)

    loader = instaloader.Instaloader(
        download_pictures=True,
        download_videos=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern=""
    )

    loader.download_profile(username, profile_pic=False, fast_update=True)

    for root, _, files in os.walk(username):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                shutil.copy(os.path.join(root, file), os.path.join("photos", file))
    shutil.rmtree(username, ignore_errors=True)
