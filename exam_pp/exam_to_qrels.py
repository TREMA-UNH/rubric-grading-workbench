
from dataclasses import dataclass
from parse_qrels_runs_with_text import *
from typing import Path, List
from question_types import QuestionPromptWithChoices, parseQueryWithFullParagraphs, QueryWithFullParagraphList
from question_types import *


@dataclass
class QrelEntry:
    query_id:str
    paragraph_id:str
    grade:int

    def format_qrels(self):
        return ' '.join([self.query_id, '0', self.paragraph_id, str(self.grade)])+'\n'

def exam_to_qrels_files(exam_input_file:Path, qrel_out_file:Path):
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)
    qrel_entries = exam_to_qrels(query_paragraphs)
   
    write_qrel_file(qrel_out_file, qrel_entries)

def write_qrel_file(qrel_out_file, qrel_entries):
    '''Use to write qrels file'''
    with open(qrel_out_file, 'wt', encoding='utf-8') as file:
        file.writelines(entry.format_qrels() for entry in qrel_entries)
        file.close()

def exam_to_qrels(query_paragraphs:List[QueryWithFullParagraphList])->List[QrelEntry]:
    '''workhorse to convert exam-annotated paragraphs into qrel entries.
    load input file with `parseQueryWithFullParagraphs`
    write qrels file with `write_qrel_file`
    or use convenience function `exam_to_qrels_file`
    '''
    qrel_entries:List[QrelEntry] = list()
    
    for queryWithFullParagraphList in query_paragraphs:
        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.exam_grades:
                exam_grade = para.exam_grades[0]
                numCorrect = len(exam_grade.correctAnswered)
                qrel_entries.append(QrelEntry(query_id=query_id, paragraph_id=para.paragraph_id, grade=numCorrect))
    return qrel_entries


def main():
    import argparse

    desc = f'''Convert paragraphs with exam annotations to a paragraph qrels file. \n
              The judgment level is the number of correctly answerable questions. \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             '''
    

    # Create the parser
    # parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser = argparse.ArgumentParser(description="Convert paragraphs with exam annotations to a paragraph qrels file."
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of anserable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    # Add an optional output file argument
    parser.add_argument('-o', '--output', type=str, metavar="FILE", help='Output QREL file name', default='output.qrels')

    # Parse the arguments
    args = parser.parse_args()    
    exam_to_qrels_files(exam_input_file=args.exam_annotated_file, qrel_out_file=args.output)


if __name__ == "__main__":
    main()
