import os
import base64
import time
from PIL import Image
import io
from gradio_client import Client

def resize_and_save_image(image_path):
    """Redimensionne l'image si nécessaire et la sauvegarde"""
    # Créer le dossier temp s'il n'existe pas
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    with Image.open(image_path) as img:
        # Redimensionner si l'image est trop grande
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
            print(f"Image redimensionnée à {new_size}")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Sauvegarder l'image temporaire dans le dossier temp
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        temp_path = os.path.join(temp_dir, f"{base_name}_temp.jpg")
        img.save(temp_path, format='JPEG', quality=95)
        return temp_path

def send_request(image_path):
    """Envoie l'image à l'API Gradio"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Préparer l'image
        temp_image_path = resize_and_save_image(image_path)
        
        # Créer le client Gradio
        client = Client("http://127.0.0.1:8080/")
        
        print(f"\n Envoi de la requête pour {base_name}...")
        
        # Paramètres par défaut selon la documentation
        result = client.predict(
            caption="",  # Pas de prompt texte
            image=temp_image_path,
            steps=40,
            guidance_scale=5.5,
            seed=1234,
            octree_resolution="512",
            check_box_rembg=True,
            api_name="/generation_all"
        )
        
        print("Génération terminée!")
        print("\nRéponse complète de l'API:")
        print(f"Nombre d'éléments dans result: {len(result)}")
        for i, item in enumerate(result):
            print(f"\nÉlément {i}:")
            print(f"Type: {type(item)}")
            print(f"Contenu: {item}")
        
        # Le résultat contient maintenant 4 éléments:
        # 0: white_mesh.glb dict
        # 1: textured_mesh.glb dict
        # 2: white_mesh preview HTML
        # 3: textured_mesh preview HTML
        white_mesh_dict, textured_mesh_dict, _, _ = result
        
        print(f"Type de textured_mesh_dict: {type(textured_mesh_dict)}")
        print(f"Contenu de textured_mesh_dict: {textured_mesh_dict}")
        
        # Extraire le chemin du fichier GLB texturé du résultat
        if isinstance(textured_mesh_dict, dict):
            print("Clés disponibles:", list(textured_mesh_dict.keys()))
            if 'value' in textured_mesh_dict:
                glb_path = textured_mesh_dict['value']
            else:
                print("La clé 'value' n'a pas été trouvée dans le résultat")
                return False
        else:
            glb_path = textured_mesh_dict

        print(f"Chemin du fichier GLB texturé: {glb_path}")
        
        # Créer le dossier output_models s'il n'existe pas
        output_dir = "output_models"
        os.makedirs(output_dir, exist_ok=True)
        
        # Renommer le fichier GLB avec notre nom personnalisé dans output_models
        output_filename = os.path.join(output_dir, f"{base_name}_3d.glb")
        
        # Vérifier si des fichiers de texture existent dans le même dossier que le GLB
        glb_dir = os.path.dirname(glb_path)
        texture_files = [f for f in os.listdir(glb_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and 'texture' in f.lower()]
        
        if os.path.exists(glb_path):
            os.rename(glb_path, output_filename)
            print(f" Modèle 3D sauvegardé dans {output_filename}")
            
            # Copier les textures si elles existent
            for texture in texture_files:
                texture_path = os.path.join(glb_dir, texture)
                texture_output = os.path.join(output_dir, f"{base_name}_{texture}")
                import shutil
                shutil.copy2(texture_path, texture_output)
                print(f" Texture copiée: {texture_output}")
        
        # Nettoyer les fichiers temporaires
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        # Nettoyer le dossier temporaire de Gradio
        if os.path.exists(glb_dir):
            import shutil
            shutil.rmtree(glb_dir)
            
        return True
    except Exception as e:
        print(f" Erreur: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def batch_process_images(input_folder, single_test=False):
    """Traite les images du dossier"""
    if not os.path.exists(input_folder):
        print(" Le dossier spécifié n'existe pas.")
        return
    
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        print(" Aucune image trouvée dans le dossier.")
        return
    
    print(f"\n Traitement de {len(images)} images...")
    
    if single_test:
        # Ne traiter que la première image pour tester
        test_image = os.path.join(input_folder, images[0])
        print(f"\nTest avec l'image: {images[0]}")
        send_request(test_image)
    else:
        # Traiter toutes les images
        for i, image in enumerate(images, 1):
            image_path = os.path.join(input_folder, image)
            print(f"\n[{i}/{len(images)}] Traitement de {image}")
            send_request(image_path)
            time.sleep(1)  # Petite pause entre chaque image

if __name__ == "__main__":
    # Dossier contenant les images à traiter
    input_folder = "input_images"
    batch_process_images(input_folder)