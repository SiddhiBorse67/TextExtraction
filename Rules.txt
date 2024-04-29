1.  This model works by comparing the image (suppose admission form), first we must provide the path of blank form in cv2.imread() function. 
2. The input file must be in *.jpg or *.png extensions. Image with *.pdf and other extensions are not allowed. 
3. The image should be clear enough to detect text from it. 
4.Before using Connect this model to database if you want to store content in database. Otherwise it displaying the content on console as well as storing inside the file (Make file in same directory with extension *.csv)  
5. Before running install all the necessary libraries.
6. Region.py file is use to find the co-ordinates of each field which I have written at start of program
   i) run the Region.py file one by one select start and end point of field after this click on top border of opened window and clicked s(from keyboard).
