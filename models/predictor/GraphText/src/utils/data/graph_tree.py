import dgl
from omegaconf import DictConfig
import pandas as pd
import json
from copy import deepcopy
from utils.pkg.dict2xml import dict2xml
import re


class GraphTree:
    def __init__(self, data, df, center_node, subg_nodes, hierarchy, name_alias, style='xml', label=None):
        self.df = df
        self.text = data.text
        self.style = style
        self.subg_nodes = subg_nodes
        self.center_node = center_node
        self.hierarchy = hierarchy
        self.label = label
        self.tree_dict = {}
        self.encode_df = pd.DataFrame()
        if len(self.df):
            grouped_tree_df = self.df.groupby(hierarchy).agg({'nodes': list}).reset_index()
            grouped_tree_df['center_node'] = center_node
            self.prompt = self.traverse(data, grouped_tree_df, hierarchy, name_alias)
        else:
            self.prompt = ''
        return

    def traverse(self, data, grouped_tree_df, hierarchy, name_alias):
        self.continuous_row_dict = []

        # ! Convert the grouped DataFrame to a nested dictionary (to be traversed)

        def extract_indices(s):
            # Find all occurrences of the pattern "<CONT_FIELD-{index}>"
            matches = re.findall(r'<CONT_FIELD-(\d+)>', s)

            # Convert the extracted indices to integers and return as a list
            return [int(match) for match in matches]

        cont_field_str_template = '<CONT_FIELD-{index}>'
        for index, row in grouped_tree_df.iterrows():
            current_dict = self.tree_dict  # Pointer to the current dictionary

            # Traverse through hierarchy levels
            for level in hierarchy[:-1]:
                level_key = name_alias.get(row[level], row[level])

                if level_key not in current_dict:
                    current_dict[level_key] = {}

                current_dict = current_dict[level_key]  # Move pointer down

            # Final hierarchy level for leaf nodes
            field = row[hierarchy[-1]]

            if row.attr_type in data.in_cont_fields:
                if len(row.nodes) > 0:
                    current_dict[name_alias.get(field, field)] = cont_field_str_template.format(index=index)
            else:  # Text
                content = [data.text.iloc[_][row.attr_type] for _ in row['nodes'] if _ != -1]
                if isinstance(content, list):
                    content = [_ for _ in content if _ != 'NA']
                if len(content) > 0:
                    current_dict[name_alias.get(field, field)] = str(content)

        if self.style == 'xml':
            graph_str = dict2xml(self.tree_dict, wrap="information", indent="\t")
            # Post-processing continuous feature
            cont_indices = extract_indices(graph_str)
            if len(cont_indices) > 0:  # Process continuous feature to encode
                # ! Store df to encode (only continuous methods needs further encoding)
                self.encode_df = grouped_tree_df.loc[cont_indices]
                for index, row in self.encode_df.iterrows():
                    placeholder = "".join([f"<{row.attr_type} emb>" for _ in row['nodes']])
                    placeholder = f"<{row.attr_type}> {placeholder} </{row.attr_type}>"
                    graph_str = graph_str.replace(cont_field_str_template.format(index=index), placeholder)
            assert len(extract_indices(graph_str)) == 0
        else:
            raise ValueError(f'Unsupported prompt style {self.style}')
        return graph_str

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt


if __name__ == "__main__":
    # Create a sample DataFrame
    # df = pd.DataFrame({
    #     'node_id': [0, 1, 2, 3, 4, 5] + [1, 3, 5],
    #     'SPD': [0, 1, 1, 2, 2, 2] + [1, 2, 2],
    #     'feature_type': ['x'] * 6 + ['y'] * 3
    # })
    #
    # # Display the nested dictionary
    # print(f"Nested Dictionary: {tree_as_dict}\nFlattened graph:")
    # print()
    # print(json.dumps(tree_as_dict, indent=4))

    df = pd.DataFrame({
        'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'SPD': [10, 20, 10, 30, 20, 10, 30, 30],
        'feature_type': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'C'],
        'attribute1': ['X', 'X', 'Y', 'Z', 'W', 'Y', 'Z', 'V'],
        'attribute2': [100, 200, 100, 300, 200, 100, 300, 300]
    })

    # Define your hierarchy and aggregation
    hierarchy = ['SPD', 'feature_type']

    agg_dict = {col: 'first' for col in df.columns if col not in hierarchy}
    agg_dict['node_id'] = list  # Assuming you want to group 'node_id'

    # Group the DataFrame
    grouped_df = df.groupby(hierarchy).agg(agg_dict).reset_index()

    print(grouped_df)
