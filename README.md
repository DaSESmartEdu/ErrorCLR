# ErrorCLR: Semantic Error Classification, Localization and Repair for Introductory Programming Assignments

In this work, we present a new dataset COJ2022 of student buggy programs and propose a two-stage model ErrorCLR for semantic error classification, localization and repair. COJ2022 can support the development of methods for all the three tasks, thereby enriching the ways of automated feedback for programming training.

**If you find this work useful for your research, please consider citing our paper.**
```@inproceedings{HanWL23,
  author       = {Siqi Han and
                  Yu Wang and
                  Xuesong Lu},
  title        = {ErrorCLR: Semantic Error Classification, Localization and Repair for
                  Introductory Programming Assignments},
  booktitle    = {Proceedings of the 46th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2023, Taipei,
                  Taiwan, July 23-27, 2023},
  pages        = {1345--1354},
  year         = {2023},
  url          = {https://doi.org/10.1145/3539618.3591680},
  doi          = {10.1145/3539618.3591680}
}
```
## COJ2022 Dataset

Our dataset includes the following files:

- submission_info.csv
- error_info.csv
- test_case.csv
- error_types_info.md
- source_codes.zip


#### Metadata at submission_info.csv

This file contains the information of all submissions (error codes and template codes):

| name of column | data type | description                            |
| -------------- | --------- | -------------------------------------- |
| ID             | string    | unique id of the submission            |
| Problem_ID     | string    | anonymized id of the problem           |
| Create_Time    | timestamp | time of the created submission         |
| User_ID        | string    | anonymized user id of the submission   |
| Result         | int       | evaluation result of the submission    |
| Language       | string    | programming language of the submission |

- submission result
  - COMPILE_ERROR = -2
  - WRONG_ANSWER = -1
  - SUCCESS = 0
  - CPU_TIME_LIMIT_EXCEEDED = 1
  - REAL_TIME_LIMIT_EXCEEDED = 2
  - MEMORY_LIMIT_EXCEEDED = 3
  - RUNTIME_ERROR = 4
  - SYSTEM_ERROR = 5



#### Metadata at error_info.csv

This file contains the information of all error lines:

| name of column | data type | description                               |
| -------------- | --------- | ----------------------------------------- |
| ID             | string    | unique id of the submission               |
| User_ID        | string    | anonymized user id of the submission      |
| Problem_ID     | string    | anonymized id of the problem              |
| Buggy_Line     | string    | text of the buggy line                    |
| Type           | string    | error type of the line (4 categories)     |
| Sub_Type       | string    | error subtype of the line (16 categories) |
| Line_ID        | int       | position / line number of the line        |
| Repaired_Line  | string    | text of the repaired line                 |


#### Metadata at test_case.csv

This file contains the information of all test cases per problem:

| name of column | data type | description                  |
| -------------- | --------- | ---------------------------- |
| Problem_ID     | string    | anonymized id of the problem |
| Std_In         | string    | text of the standard input   |
| Std_Out        | string    | text of the standard output  |


#### Metadata at error_types_info.md

This file contains the description and examples of the error types and sub_types. Currently, we identify the error lines with one kind of subtype. If their contains multiple errors in one line, this will be assigned with the ``undefined`` type. In the future, we will improve our methods such as multi-labels or automatic labeling. We are now exploring the automatic labeling with AST node-level differences.



## ErrorCLR

ErrorCLR is a novel two-stage method to solve semantic error classification, localization and repair as dependent tasks for introductory programming assignments. 

### Requirements

Install all the dependent packages in `requirement.txt`  via pip.

### Stage 1: Error classification and localization

- `cd Stage1`
- Prepare [JOERN](https://github.com/joernio/joern) to generate CFGs of programs
- Run `process.py` to get graph input
- Use `train.py` to pre-train graph matchability and `finetune.py` to fine-tune error classification and location

### Stage 2: Error repair

- `cd Stage2`
- Download pre-trained checkpoints of codeT5 from https://github.com/salesforce/CodeT5
- View ``mask_utils.py`` for our span masking and commenting strategy to prepare for inputs
  - API: create_mask(*errorLine*, *repairLine*, *type*, *subtype*, *mask_cnt=0*, *target_line=""*)
- `cd sh` and `python -u run_exp.py --task refine --sub_task oj`
