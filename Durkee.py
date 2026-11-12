
import warnings
warnings.simplefilter('ignore')
import numpy as np

import itertools
import networkx as nx
import pandas as pd

from collections import defaultdict


class MyDurkee():

    def __init__(self, verbose=False):
        self.verbose=verbose
        pass
    
    def create_pairwise_agreement_matrix(self, r1_mask_array, r2_mask_array, bglabel=None):
        
        im1 = r1_mask_array
        im2 = r2_mask_array
        
        counts_1 = np.bincount(im1.ravel('K')).astype(np.uint32)
        counts_2 = np.bincount(im2.ravel('K')).astype(np.uint32)
        
        shape = (len(counts_1), len(counts_2))
        
        linear_ids = np.ravel_multi_index((im1, im2), shape).astype(np.uint32)
        intersection = np.bincount(linear_ids.ravel(), minlength=np.prod(shape)).astype(np.uint32)
        intersection = intersection.reshape(shape)
        
        union = counts_1[:, np.newaxis] + counts_2 - intersection
        
        iou_matrix = np.zeros(shape, np.single)
        np.divide(intersection, union, out=iou_matrix, where=(union != 0))
        
        # sort out non-existing element-ids
        cell_ids1 = np.flatnonzero(counts_1).astype(np.uint32)
        cell_ids2 = np.flatnonzero(counts_2).astype(np.uint32)

        if bglabel is not None:
            ind_1 = np.where(cell_ids1 == bglabel)[0]
            ind_2 = np.where(cell_ids2 == bglabel)[0]

            # Handle rows (cell_ids1)
            if ind_1.size > 0:
                idx1 = ind_1[0]
                cell_ids1 = np.concatenate((cell_ids1[:idx1], cell_ids1[idx1+1:]))
                iou_matrix = np.delete(iou_matrix, idx1, axis=0)

            # Handle columns (cell_ids2)
            if ind_2.size > 0:
                idx2 = ind_2[0]
                cell_ids2 = np.concatenate((cell_ids2[:idx2], cell_ids2[idx2+1:]))
                iou_matrix = np.delete(iou_matrix, idx2, axis=1)  

        # also in the cost matrix: remove non existing object-ids
        #a_mat = iou_matrix[np.ix_(cell_ids1, cell_ids2)]
        a_mat = iou_matrix
        
        
        r1_cells = cell_ids1
        r2_cells = cell_ids2
        
        if len(r1_cells) > 0 and len(r2_cells) > 0:
        
            idx_row = np.argmax(a_mat, axis=0)
            aR = np.zeros_like(a_mat)
            aR[idx_row, np.arange(a_mat.shape[1])] = a_mat[idx_row, np.arange(a_mat.shape[1])]

            idx_col = np.argmax(a_mat, axis=1)
            aC = np.zeros_like(a_mat)
            aC[np.arange(a_mat.shape[0]), idx_col] = a_mat[np.arange(a_mat.shape[0]), idx_col]
        
            aggmat = np.sqrt(aR * aC)
        else:
            aggmat = a_mat.copy()

        agg = {
            'R1count': len(r1_cells), 
            'R2count': len(r2_cells), 
            'AgreementMatrix': aggmat
        }
        
        #print(aggmat)
        
        return agg

    
    def create_match_matrix_fast(self, agreement_dicts, M, thresholds=[0.25, 0.5, 0.75]):
        results_by_threshold = {}

        for th in thresholds:
            G_multi = nx.Graph()

            for pair_key, compdict in agreement_dicts.items():
                parts = pair_key.split('_')
                R1, R2 = parts[0], parts[1]

                agg = compdict['AgreementMatrix']
                n_rows, n_cols = agg.shape

                row_idxs, col_idxs = np.nonzero(agg >= th)
                weights = agg[row_idxs, col_idxs]

                all_rows = [f'{R1}-{i}' for i in range(n_rows)]
                all_cols = [f'{R2}-{i}' for i in range(n_cols)]
                G_multi.add_nodes_from(all_rows)
                G_multi.add_nodes_from(all_cols)

                edges = [
                    (f'{R1}-{r}', f'{R2}-{c}', {'weight': float(w)})
                    for r, c, w in zip(row_idxs, col_idxs, weights)
                ]
                G_multi.add_edges_from(edges)

            if self.verbose:
                # Counter for components with multiple of the same base name
                count_with_duplicates = 0

                # Iterate through each connected component
                for component in nx.connected_components(G_multi):
                    base_name_counts = defaultdict(int)

                    for node in component:
                        # Extract the base name by splitting at '-'
                        base_name = node.split("-")[0]
                        base_name_counts[base_name] += 1

                    # Check if any base name appears 2 or more times in this component
                    has_duplicates = any(count > 1 for count in base_name_counts.values())

                    if has_duplicates:
                        count_with_duplicates += 1

                print(
                    f"Number of connected components with multiple of the same MX: {count_with_duplicates}"
                )
                
            consensus_objects = nx.number_connected_components(G_multi) 
            total_calls = G_multi.number_of_nodes()

            results_by_threshold[th] = {'C': total_calls, 'O': consensus_objects, 'Nr': M}

        return results_by_threshold

    def create_match_matrix(self, agreement_dicts, M, thresholds=[0.25, 0.5, 0.75]):
        results_by_threshold = {}

        for th in thresholds:

            G_multi = nx.Graph()
            
            for pair_key, compdict in agreement_dicts.items():
                R1, R2 = pair_key.split('_')[0], pair_key.split('_')[1]
                
                agg = compdict['AgreementMatrix']
                agg = np.where(agg >= th, agg, 0)

                n_rows, n_cols = agg.shape
                rows = [f'{R1}-{i}' for i in range(n_rows)]
                cols = [f'{R2}-{i}' for i in range(n_cols)]

                agg = pd.DataFrame(agg, columns=cols, index=rows)
                
                
                # Step 1: Add all row and column identifiers as nodes first
                G_multi.add_nodes_from(agg.index)
                G_multi.add_nodes_from(agg.columns)

                # Step 2: Iterate and add edges where values > 0
                for r in agg.index:
                    for c in agg.columns:
                        if agg.loc[r, c] > 0:
                            G_multi.add_edge(r, c, weight=agg.loc[r, c])

                            
            consensus_objects = len(list(nx.connected_components(G_multi)))
            total_calls = G_multi.number_of_nodes()

            results_by_threshold[th] = {'C': total_calls, 'O': consensus_objects, 'Nr': M}

        return results_by_threshold
    
    
    def get_metric(self, results_per_th):
        obj = results_per_th['O']
        Nr = results_per_th['Nr']
        preds = results_per_th['C']

        if self.verbose:
            print(f'M={Nr}, C={preds}, O={obj}')
        
        MR_JI = (preds-obj)/((Nr-1)*obj)
        MR_SI = Nr*(preds-obj)/(((Nr-1)**2)*obj+preds-obj) #Nr*MR_JI/(Nr-1+MR_JI)

        return MR_JI, MR_SI
    
    def do_it(self, masks, bglabel=None, verbose=False):
        
        tmp_masks = {}
        
        for i,m in enumerate(masks):
            tmp_masks[f'M{i}'] = m

        if self.verbose:
            print('Mask dict created')
            
        agreements = {}
        for r1, r2 in itertools.combinations(tmp_masks.keys(), 2):
            pair_key = f"{r1}_{r2}"
            agreements[pair_key] = self.create_pairwise_agreement_matrix(tmp_masks[r1], tmp_masks[r2], bglabel=bglabel)
        
            if self.verbose:
                print(f'Agreement {r1}-{r2} finished.')
        
        match_results = self.create_match_matrix_fast(agreements, len(masks))
        
        if self.verbose:
            print('Match dict created')
            
        scores = {}
        
        # do per th
        for th in [0.25, 0.5, 0.75]:
          #  print(th)
            ji, si = self.get_metric(match_results[th])
          #  print(th, ji, si)
            scores[th] = {'JI': ji, 'SI': si}
        return scores
        
        
        
        
        
        
        
        
        
        
        