import os, shutil, glob
folder = '/home/fede/Desktop/cross_fold_segmented/0/' #root folder from which all subfolders cancel pngs

for img in glob.glob(folder + "*.png"):
  os.remove(img)

for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    if os.path.isdir(file_path): #shutil.rmtree(file_path)
       for img in glob.glob(file_path + "/*.png"):
           os.remove(img)
        
           
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    for the_file in os.listdir(folder):
       file_path = os.path.join(folder, the_file)
       if os.path.isdir(file_path): #shutil.rmtree(file_path)
          for img in glob.glob(file_path + "/*.png"):
              os.remove(img)           

#for the_file in os.listdir(folder):
#    file_path = os.path.join(folder, the_file)
#    try:
#        if os.path.isfile(file_path):            
#            os.unlink(file_path)
#        #elif os.path.isdir(file_path): shutil.rmtree(file_path) #if also want to remove subdirectories, uncomment the elif statement
#    except Exception as e:
#        print(e)


#import os
#path = '/home/zanetti/Scrivania/My_Programs/prova'
#def scandirs(path):
#    for root, dirs, files in os.walk(path):
#        for currentFile in files:
#            print "processing file: " + currentFile
#            exts = ('.png')
#            if any(currentFile.lower().endswith(ext) for ext in exts):
#                os.remove(os.path.join(root, currentFile))
