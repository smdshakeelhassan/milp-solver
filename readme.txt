Please install the required packages in a fresh python environment using the command

pip install -r requirements.txt

Change the matrix A and vectors c and values in the files mat_A.txt, vec_c.txt and vec_b.txt respectively.
The values are space separated entries and each row of the matrix has to be entered in a new line.

Run the code by using the following command:

python milp_solver.py

The output will show the best objective and best decision variables in the first line and then print out the enumeration
tree.

Leaf nodes in the tree are decorated in **Node** format and infeasible nodes are marked as Infeasible.

An IDE like PyCharm and small input values is recommended for pretty tree outputs. A sample output for the problem with
current input values is provided as sample_output.png