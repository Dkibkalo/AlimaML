import os
import face_recognition
import shutil

def sort_faces(input_folder: str, output_folder: str, tolerance: float = 0.6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    known_faces = []
    face_groups = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filepath = os.path.join(input_folder, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"No faces found in {filename}")
            continue

        face_encoding = encodings[0]

        match_index = None
        for i, known_encoding in enumerate(known_faces):
            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
            if match[0]:
                match_index = i
                break

        if match_index is not None:
            group_id = match_index
        else:
            group_id = len(known_faces)
            known_faces.append(face_encoding)
            face_groups.append([])

        person_folder = os.path.join(output_folder, f"person_{group_id}")
        os.makedirs(person_folder, exist_ok=True)

        shutil.copy(filepath, os.path.join(person_folder, filename))
        face_groups[group_id].append(filename)

    print("Finished grouping.")
