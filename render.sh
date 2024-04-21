Folder_A="./output"  
for file_a in ${Folder_A}/*
do  
    temp_file=`basename $file_a`
    python render.py -m ${Folder_A}/$temp_file
done

