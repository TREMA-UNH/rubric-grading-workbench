from typing import List, Dict, Optional
from pathlib import Path

from pylatex import Document, Tabu, MultiColumn, MultiRow, NoEscape, Command, Package, Tabular, Section
from pylatex.utils import bold, italic

class TablePrinter():
    def __init__(self):
        self.doc = Document("standalone")
        self.doc.packages.append(Package('booktabs'))
        self.doc.packages.append(Package('amsmath'))
        self.doc.packages.append(Package('rotating'))  # Add the booktabs package
        self.doc.packages.append(Package('xcolor', options=['table']))  # highlight table 
        self.doc.preamble.append(NoEscape(r'\setlength{\fboxsep}{1pt}'))
        
    def export(self, table_file_name:Path):
        # Get LaTeX representation as a string
        latex_code = self.doc.dumps()

        # Print the LaTeX code
        # print(latex_code)

        # Optionally, save the LaTeX code to a file
        with open(table_file_name, 'w') as file:
            file.write(latex_code)
            file.close()
            print(f"correlation table written to {table_file_name}.") 
        
    def add_section(self, heading:str):
        with self.doc.create(Section(heading)):
            self.doc.append(" ")

        
    def add_new_paragraph(self):
        self.doc.append(NoEscape(r'\par'))  # Paragraph break

    def add_table(self,  counts:Dict[str,Dict[str,int]], kappa:Dict[str,float]
                  , judgments_header:List[str], label_header:List[str]
                  , judgment_title:Optional[str]="Assessors", label_title:Optional[str]="GRADED"
                  , label_to_judgment_kappa:Optional[Dict[str,str]]=None):
        # Find the maximum values for each column
        max_col_values = {judgment: max(counts[label][judgment] for label in label_header) for judgment in judgments_header}
        max_row_values = {label: max(counts[label][judgment] for judgment in judgments_header) for  label in label_header}

        # Create a Document instance

        # Begin the tabu environment with specified column alignments
        with self.doc.create(Tabu(f'll{"".join(["c" for _ in judgments_header])}lr', booktabs=True)) as table:

            rotated_label_title = NoEscape(r'\begin{sideways}'+label_title+r'\end{sideways}')
            multirow_title = MultiRow(len(label_header)+2, data=rotated_label_title)

            # Add the header row
            table.add_row((multirow_title, bold('Label'), 
                        MultiColumn(len(judgments_header), align='c', data=bold(judgment_title)), 
                        bold("Total"),
                        bold("Cohen's ") + NoEscape(r'$\boldsymbol{\kappa}$')))
            
            # Add the cmidrule
            table.append(NoEscape(r'\cmidrule(l@{\tabcolsep}){3-' + str(2 + len(judgments_header)) + '}'))
            
            # Second header row
            table.add_row(['', '', *judgments_header,'', ''])
            table.append(NoEscape(r'\cmidrule(l@{\tabcolsep}){1-' + str(4 + len(judgments_header)) + '}'))

            # Data rows
            for l in label_header:
                # Moved the label_header two rows up.
                # row_title_element = multirow_title if label_header.index(l) == 0 else ''
                row_title_element =''

                def format_counts_bold(l:str, j:str):
                    cell = str(counts[l][j])
                    # if counts[l][j] == max_row_values[l]:   # highest count per row
                    #     return bold(cell)
                    if counts[l][j] == max_col_values[j]:  # highest count per column
                        return bold(cell)
                    else:
                        return cell

                def format_counts(l:str, j:str):
                    cell = format_counts_bold(l,j)
                    if label_to_judgment_kappa is not None and (label_to_judgment_kappa.get(l,None) == j):
                        return NoEscape(r'\fbox{'+cell+'}')
                    else:   
                        return cell


                def format_kappa(l:str):
                    if (l in kappa) and (kappa[l]is not None):
                        return f'{kappa[l]:.2}'  
                    else:
                        return ''

                def format_row_total(l:str):
                    total = sum( (counts[l][j] for j in judgments_header) )
                    return str(total)


                formatted_scores = [format_counts(l,j) for j in judgments_header]
                table.add_row([row_title_element, l, *formatted_scores, format_row_total(l), format_kappa(l)])




def print_table( table_file_name:Path, counts:Dict[str,Dict[str,int]], kappa:Dict[str,float], judgments_header:List[str], label_header:List[str], judgment_title:str="Assessors", label_title:str="GRADED"):
    printer = TablePrinter()
    printer.add_table(counts=counts, kappa=kappa, judgments_header=judgments_header, label_header=label_header, judgment_title=judgment_title, label_title=label_title)
    printer.export(table_file_name)


def print_example_table():
    # Define the variable headers for the judgments and labels
    judgments_header = ['3', '2', '1', '0']
    label_header = ['Relevant1', 'Non-relevant', 'Relevant2']

    # Data dictionaries
    counts = {
        'Relevant1': {'3': 89, '2': 65, '1': 48, '0': 16},
        'Non-relevant': {'3': 11, '2': 35, '1': 52, '0': 84},
        'Relevant2': {'3': 96, '2': 93, '1': 79, '0': 42}
    }
    kappa = {'Relevant1': 0.40, 'Non-relevant': None, 'Relevant2': 0.49}
    print_table(counts=counts, kappa=kappa, judgments_header=judgments_header, label_header=label_header, judgment_title="Judgments", label_title="GRADED")
        

if __name__ == "__main__":
    print_example_table()
