=========================================================================================
REQUIREMENTS:

This system requires networkx library as dependency

pip install networkx 

=========================================================================================
To run the system execute following command 

python main.py path_to_inkml 'segment/perfect' [bonus]

"path_to_inkml" should be a directory containing all the inkml files.
The program does not recursively look for directory in directory.

Example : 
python main.py path_to_inkml segment

OR

python main.py path_to_inkml perfect

The bonus parser is only for perfectly segmented symbols
To run for bonus, execute following command

python main.py path_to_inkml perfect

"perfect" flag is for running parsing on perfectly segmented symbols
"segment" flag is for running parsing on strokes level input

=========================================================================================

Output files

At the end of execution, a folder will be generated named : input_directory_"output_lg"
This folder will contain all the .lg files for the inkml files given as input 

=========================================================================================

Source code

All the python files should be placed in same directory

=========================================================================================