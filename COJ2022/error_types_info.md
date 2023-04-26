# Error rules of COJ2022

We totally identify 4 types of semantic errors covering 15 sub-types in the programs. For each type, we select the most common sub-type errors and classify them. Some of the sub-type errors appears in multiple types due to the frequency and error habits of students.

### Control

**errors pertaining to a control statement**

- ControlMisuse
  - Description: Misuse of control actions (while / for/ if / break / continue / else / else if / return)
  - Examples
    - **BEFORE:** ``break;``    **AFTER:** ``continue;``
    - **BEFORE:** ``if(m > n)``    **AFTER:** ``while(m > n)``
    - **BEFORE:** ``else printf(" ")``    **AFTER:** `` ``
- ConditionMissing
  - Description: Missing of the control conditions and control statements
  - Examples
    - **BEFORE:** ``if(a > 0)``    **AFTER:** ``if(a > 0 && b != 1)``
    - **BEFORE:** `` ``    **AFTER:** ``return 0;``
    - **BEFORE:** ``else putchar(' ')``    **AFTER:** ``else if(j < m-d) putchar(' ')``

- WrongIndex
  - Description: Error about the index in arrays no matter the variables/literals/operators...
  - Examples
    - **BEFORE:** ``if(b[i][j+m-n]==a[i][j])``    **AFTER:** ``if(b[i][(j+m)%n]==a[i][j])``

- WrongOperator
  - Description: Error about the operators (arithmetic / relational / logical ops)
  - Examples
    - **BEFORE:** ``if(i<n)``    **AFTER:** ``if(i<=n)``
- LiteralMisuse
  - Description: Misuse of the literals (number / string / char / constant)
  - Examples
    - **BEFORE:** ``for(i = 0; i < 30; i++)``    **AFTER:** ``for(i = 1; i < 30; i++)``
    - **BEFORE:** ``return NULL;``    **AFTER:** ``return "";``
    - **BEFORE:** ``while((ch[i]=getchar())!='\n')``    **AFTER:** ``while((ch[i]=getchar())!=EOF)``

- VariableMisuse
  - Description: Misuse of the variables
  - Examples
    - **BEFORE:** ``for(j = 0; j < n-1; j++)``    **AFTER:** ``for(j = 0; j < i-1; j++)``

- OffByOneError
  - Description: Error about the differences of +1/-1 on variables
  - Examples
    - **BEFORE:** ``while(sum != n)``    **AFTER:** ``while(sum != n+1)``

### Function

**errors pertaining to function call and declaration**

We include input/output statement(printf / scanf / getchar / cin / ... ) into the function category.

- FunctionMisuse
  - Description: Misuse of the functions calls
  - Examples
    - **BEFORE:** ``y = abs(x);``    **AFTER:** ``y = floor(x);``
    - **BEFORE:** ``system("pause";)``    **AFTER:** `` ``
    - **BEFORE:** ``printf("%.4f", fabs(ans))``    **AFTER:** ``printf("%.4f", ans)``

- FunctionMissing
  - Description: Missing of the function calls
  - Examples
    - **BEFORE:** `` ``    **AFTER:** ``getchar();``
    - **BEFORE:** ``p = ps+pt;``    **AFTER:** `` p = round(ps+pt)``
    - **BEFORE:** ``printf("%.4f", fabs(ans))``    **AFTER:** ``printf("%.4f", ans)``

- ParameterTypeMisuse
  - Description:  Misuse of the type of parameter in function declaration and calls
  - Examples
    - **BEFORE:** `` scanf("%d", a);``    **AFTER:** ``scanf("%d", &a);``
    - **BEFORE:** ``if(strcmp(*t[i],*t[m])>=0)``    **AFTER:** `` if(strcmp(t[i],t[m])>=0)``

- FormatStringMisuse
  - Description:  Misuse of the format strings in input/output statements
  - Examples
    - **BEFORE:**  ``printf("%f", a); ``  **AFTER:** ``printf("%.2f", a);``
    - **BEFORE:**  ``cout<<"+"; ``   **AFTER:** ``cout<<" + ";``
- WrongIndex
  - Description:  Error about the index in arrays no matter the variables/literals/operators...
  - Examples
    - **BEFORE:** ``if(compare(a[r], a[n])){``  **AFTER:** ``if(compare(a[r], a[max])){``

### Declaration

**errors pertaining to variable declaration**

- DataTypeMisuse
  - Description:  Misuse of data type on variables
  - Examples
    - **BEFORE:** ``int sum = 0; ``  **AFTER:** ``long sum = 0; ``

- WrongArraySize
  - Description:  Error about the size of declared arrays
  - Examples
    - **BEFORE:** ``int a[n+1]; ``  **AFTER:** ``int a[n+10]; ``

- InitializationMissing
  - Description:  Missing of the initial
  - Examples
    - **BEFORE:** ``int i, sum;``  **AFTER:** ``int i=0, sum=0; ``

- LiteralMisuse
  - Description: Misuse of the literals (number / string / char / constant)
  - Examples
    - **BEFORE:** ``int max_n = 8;``    **AFTER:** ``int max_n = 10;``
    - **BEFORE:** ``int dy[4] = {1,0,0,-1};``    **AFTER:** ``int dy[4] = {-1,0,0,1};``



### Expression

**errors inside a normal expression**

- WrongIndex
  - Description: Error about the index in arrays no matter the variables/literals/operators...
  - Examples
    - **BEFORE:** ``c = a[i][j] * b[j];``    **AFTER:** ``c = a[i][k] * b[k];``
- WrongOperator
  - Description: Error about the operators (arithmetic / relational / logical ops)
  - Examples
    - **BEFORE:** ``a += 1;``    **AFTER:** ``a -= 1``
- LiteralMisuse
  - Description: Misuse of the literals (number / string / char / constant)
  - Examples
    - **BEFORE:** ``ans = ans * 10 + s[i];``    **AFTER:** ````ans = ans * 16 + s[i];````
- VariableMisuse
  - Description: Misuse of the variables
  - Examples
    - **BEFORE:** ``pre = n;``    **AFTER:** ``pre = i;``
- OffByOneError
  - Description: Error about the differences of +1/-1 on variables
  - Examples
    - **BEFORE:** ``m = n/2;``    **AFTER:** ``m = n/2+1;``