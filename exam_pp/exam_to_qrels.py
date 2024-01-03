

from parse_qrels_runs_with_text import *


@dataclass
class QrelEntry:
    query_id:str
    paragraph_id:str
    grade:int

    def format_qrels(self):
        return ' '.join([self.query_id, '0', self.paragraph_id, str(self.grade)])+'\n'

def examToQrels(input_file:Path, qrel_out_file:Path):
    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(input_file)
    qrel_entries = list()
    
    for queryWithFullParagraphList in query_paragraphs:

        query_id = queryWithFullParagraphList.queryId
        paragraphs = queryWithFullParagraphList.paragraphs

        for para in paragraphs:
            if para.exam_grades:
                exam_grade = para.exam_grades[0]
                numCorrect = len(exam_grade.correctAnswered)
                qrel_entries.append(QrelEntry(query_id=query_id, paragraph_id=para.paragraph_id, grade=numCorrect))


    
    with open(qrel_out_file, 'wt', encoding='utf-8') as file:
        file.writelines(entry.format_qrels() for entry in qrel_entries)
        file.close()


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
    examToQrels(input_file=args.exam_annotated_file, qrel_out_file=args.output)


if __name__ == "__main__":
    main()