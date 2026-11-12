
import numpy as np
from scipy import sparse
import sparse as sparselib
from ortools.linear_solver import pywraplp
       
class MetricsMultiMask():
    
    def __init__(self, verbose=True):
        self.num_cells = []
        self.iou_dict = None
        self.assigned = None
        self.verbose = verbose
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    # ++++++++ BLOCK 1 ++++++++++++++++++++++++++++++++++    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    
    def register(self, masks):
        self.__init__(self.verbose)
        def sparse_indicator_matrix(mask):
            labels_a = mask.ravel('K').astype(np.uint32)

            self.num_cells.append(len(np.unique(labels_a)))
        
            num_samples = labels_a.shape[0]
            num_objects_a = labels_a.max() + 1 # + zero

            rows_a = labels_a
            cols = np.arange(num_samples, dtype=np.uint32)
            data = np.ones(num_samples, dtype=np.uint32)

            sim = sparse.csr_matrix((data, (rows_a, cols)), shape=(num_objects_a, num_samples), dtype=np.uint32)
            return sim
        
        matrices = []
        for mask in masks:
            tmp = sparselib.COO.from_scipy_sparse(sparse_indicator_matrix(mask))
            matrices.append(tmp)
            
        
        # Step 1: Build col_to_row 
        M = len(matrices)
        n_cols = matrices[0].shape[1]
        col_to_row_list = []
        for m, X in enumerate(matrices):
            col_to_row = np.full(n_cols, -1, dtype=int)
            rows = X.coords[0]
            cols = X.coords[1]
            for i, j in zip(rows, cols):
                col_to_row[int(j)] = int(i)
            col_to_row_list.append(col_to_row)

        # Step 2: Compute intersection counts
        T_intersect = {}
        for j in range(n_cols):
            rows_for_j = [col_to_row_list[m][j] for m in range(M)]
            if all(i >= 0 for i in rows_for_j):  # Only if all matrices have a 1
                key = tuple(rows_for_j)
                T_intersect[key] = T_intersect.get(key, 0) + 1

        # Step 3: Precompute row -> columns
        row_to_cols = []
        for m, X in enumerate(matrices):
            I_m = X.shape[0]
            row_to_cols_m = [set() for _ in range(I_m)]
            rows_coords = X.coords[0]
            cols_coords = X.coords[1]
            for i, j in zip(rows_coords, cols_coords):
                row_to_cols_m[int(i)].add(int(j))
            row_to_cols.append(row_to_cols_m)

        # Step 4: Compute union coverage for intersection triples only
        T_union_dict = {}
        for key in T_intersect:  # Only the sparse intersection triples
            cols_covered = set()
            for m, r_m in enumerate(key):
                cols_covered |= row_to_cols[m][r_m]
            T_union_dict[key] = len(cols_covered)

        # Step 5: Convert to sparse tensor
        if not T_union_dict:
            M = len(matrices)
            all_shapes = tuple(X.shape[0] for X in matrices)
            return sparselib.COO(coords=[[] for _ in range(M)], 
                             data=np.array([]), shape=all_shapes)

        iou_dict = {
            k: (T_intersect[k] / T_union_dict[k]) if T_union_dict[k] != 0 else None
            for k in T_intersect.keys() & T_union_dict.keys()
        }

        self.iou_dict = iou_dict
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    # ++++++++ BLOCK 2 ++++++++++++++++++++++++++++++++++    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
      
    def solve_assignment_problem(self):

        """
        Solve max-weight n-D assignment directly from dict[tuple->weight].
        """
        sparse_dict = self.iou_dict
        if not sparse_dict:
            return [], 0.0

        # A) Extract data from dict
        items = list(sparse_dict.items())
        nnz = len(items)

        # Determine ndim from first tuple (all should have same length)
        ndim = len(items[0][0])
        if self.verbose:
            print(f"Input: {nnz} nonzeros, {ndim} dimensions")

        # Filter positive weights
        positive_items = [(indices, weight) for indices, weight in items if weight > 0]
        if len(positive_items) < nnz:
            print(f"After >0 filter: {len(positive_items)} nonzeros")

        if not positive_items:
            return [], 0.0

        nnz = len(positive_items)

        # B) SCIP solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise RuntimeError("SCIP solver unavailable")

        # Binary variables - indexed by dict keys directly
        x = {}
        for i, (indices, weight) in enumerate(positive_items):
            x[indices] = solver.BoolVar(f'x_{i}_{indices}')

        # Objective: sum(weight * x[indices] for all indices)
        objective_terms = []
        for indices, weight in positive_items:
            objective_terms.append(weight * x[indices])
        solver.Maximize(solver.Sum(objective_terms))

        # Constraints: each index value <=1 per dimension
        for dim in range(ndim):
            # Group indices by their value in this dimension
            index_groups = {}
            for indices, _ in positive_items:
                val = indices[dim]
                if val not in index_groups:
                    index_groups[val] = []
                index_groups[val].append(indices)

            # Add constraint for each unique value in this dimension
            for val, indices_list in index_groups.items():
                vars_for_constraint = [x[idx] for idx in indices_list]
                solver.Add(solver.Sum(vars_for_constraint) <= 1)
        if self.verbose:
            print(f"Model built: {nnz} vars, ~{ndim*len(index_groups)} constraints")

        # Solve (5min timeout)
        solver.SetTimeLimit(300000)

        status = solver.Solve()

        selected = []
        total_weight = solver.Objective().Value()

        for indices, weight in positive_items:
            if x[indices].solution_value() > 0.5:
                selected.append((indices, float(weight)))

        self.assigned = selected
        
    def solve_assignment_problem_greedy(self):
        
        data_dict = self.iou_dict
        
        # 1. Sort entries by value in descending order
        sorted_entries = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)

        # 2. Initialize tracking sets for each position dynamically
        num_positions = len(next(iter(data_dict)))
        used_elements = [set() for _ in range(num_positions)]

        selected_values = []

        # 3. Greedy selection
        for tuple_key, value in sorted_entries:
            # Check if any element in the tuple is already used at its respective position
            is_blocked = any(elem in used_elements[i] for i, elem in enumerate(tuple_key))

            if not is_blocked:
                # Accept the value
                selected_values.append(value)
                # Mark all elements in this tuple as used for their positions
                for i, elem in enumerate(tuple_key):
                    used_elements[i].add(elem)

        self.assigned = np.flip(selected_values)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    # ++++++++ BLOCK 3 ++++++++++++++++++++++++++++++++++    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    
    def sort(self):
        self.assigned = [i[1] for i in self.assigned] # drop indices (no longer useful ? )
        self.assigned = sorted(self.assigned, reverse=False)  # sort values descending
        if len(self.assigned) == 0:
            print('Error, no overlapping masks')
            return -1
        
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    # ++++++++ BLOCK 4 ++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def calc_metric(self):
    
        tp0 = len(self.assigned)
    
        fp0s = [i - tp0 for i in self.num_cells] 
    
        ndim = len(fp0s)
        
        k = np.arange(0, tp0+1)
        
        m = (tp0 - k) / (np.sum(self.num_cells)/ndim) 
        
        t = np.append(self.assigned, 1)
        t_hat = np.insert(self.assigned, 0, 0)
 
        sf1 = np.dot((t - t_hat), m)
        
        return sf1 
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ 
    # ++++++++ UTIL FUNCTIONS +++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++    
    
    def register_wrapper(self, *masks):
        return self.register(list(masks))
        
    def clear_background(bglabel=0):
        self.assigned = {k: v for k, v in self.assigned.items() if bglabel not in k}    
    
    def do_fullrun(self, masks, bglabel=None):
        self.register(masks)
        if bglabel is not None:
            self.clear_background(bglabel)
        self.solve_assignment_problem()
        self.sort()
        return self.calc_metric()
    
    def do_fullrun_greedy(self, masks, bglabel=None):
        self.register(masks)
        if bglabel is not None:
            self.clear_background(bglabel)
        self.solve_assignment_problem_greedy()
        return self.calc_metric()

        