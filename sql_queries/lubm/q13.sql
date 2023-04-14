-- out (x) : - type (x, 'GraduateStudent'), undergraduateDegreeFrom(x, 'http://www.University567.edu').
SELECT t.subject AS x
FROM type t
JOIN undergraduateDegreeFrom udf ON t.subject = udf.subject
WHERE t.object = 'GraduateStudent'
AND udf.object = 'http://www.University567.edu';