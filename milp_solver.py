import numpy as np
from scipy.optimize import linprog
import random
import math


class BNBNode:
    def __init__(self, c, A, b, node_ub, parent=None, left=None, right=None):
        self.c = c
        self.A = A
        self.b = b
        self.node_ub = node_ub
        self.parent = parent
        self.left = left
        self.right = right
        self.objective = None
        self.decision_var = None
        self.solve_further = True

    def solve(self):
        res = linprog(self.c, A_ub=self.A, b_ub=self.b)
        try:
            self.objective = -round(res.fun,2)
            self.decision_var = [round(i,2) for i in res.x]
        except:
            self.solve_further = False

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def display(self):
        lines, *_ = self._display_aux()

        for line in lines:
            print(line)

    def _display_aux(self):
        # No child.
        if self.right is None and self.left is None:
            line = "**" + str(self.objective) + " " + str(self.decision_var) + "**"
            if line == "**None None**":
                line = "Infeasible"
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = str(self.objective) + " " + str(self.decision_var)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = str(self.objective) + " " + str(self.decision_var)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = str(self.objective) + " " + str(self.decision_var)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\'  + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


class BNBSolver:
    def __init__(self, root, upper_bound, lower_bound):
        self.root = root
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def branch_bound(self):
        best_obj = 0
        best_dec_vars = []
        waiting_nodes = [self.root]
        while waiting_nodes:
            subproblem = waiting_nodes.pop()
            subproblem.solve()

            if not subproblem.solve_further or subproblem.objective > self.upper_bound:
                continue
            else:
                is_int_sol = all([x.is_integer() for x in subproblem.decision_var])
                if is_int_sol:
                    if subproblem.objective > best_obj:
                        best_obj = subproblem.objective
                        best_dec_vars = subproblem.decision_var
                    self.lower_bound = subproblem.objective
                    for node in waiting_nodes:
                        if node.objective is not None and node.objective < self.lower_bound:
                            node.solve_further = False
                else:
                    not_int_vars = list(np.where([not x.is_integer() for x in subproblem.decision_var])[0])
                    var_index = random.choice(not_int_vars)
                    var_choice = subproblem.decision_var[var_index]
                    A_augment = np.zeros(len(subproblem.decision_var))
                    A_augment[var_index] = 1
                    node_1 = BNBNode(subproblem.c, np.concatenate([subproblem.A, [A_augment]]), np.append(subproblem.b, math.floor(var_choice)), subproblem.objective, subproblem)
                    subproblem.set_left(node_1)
                    waiting_nodes.append(node_1)

                    A_augment = np.zeros(len(subproblem.decision_var))
                    A_augment[var_index] = -1
                    node_2 = BNBNode(subproblem.c, np.concatenate([subproblem.A, [A_augment]]), np.append(subproblem.b, -math.ceil(var_choice)), subproblem.objective, subproblem)
                    subproblem.set_right(node_2)
                    waiting_nodes.append(node_2)

            self.upper_bound = max([node_1.node_ub, node_2.node_ub])

        return best_obj, best_dec_vars


if __name__ == "__main__":
    '''
    Define the Mixed Integer Linear Program from given input data, Matrix A and vectors c and b.
    maximize ζ = cT x
    S.T. A x ≤ b
         x ≥ 0
         xi ∈ Z, i ∈ {1, ···, n}
    '''
    A = np.loadtxt("mat_A.txt")
    b = np.loadtxt("vec_b.txt")
    c = -np.loadtxt("vec_c.txt")  # linprog defaults to minimization, so the weights are made negative

    if A.shape != (b.size, c.size):
        print("Please check input dimensions")
    else:
        # Augment matrix A and vector b to include non negativity constraints
        I = -np.eye(c.size)
        A = np.concatenate([A, I])
        zeros = np.zeros(c.size)
        b = np.concatenate([b, zeros])

        root_node = BNBNode(c, A, b, np.inf)
        result = BNBSolver(root_node, np.inf, -np.inf)

        print("Best Objective, Best Decision variables:")
        zeta, x = result.branch_bound()
        print(zeta, x)

        print("\n\nEnumeration Tree:\n\n")
        root_node.display()
