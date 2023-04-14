-- out (x,y) : - teacherOf ('http://www.Department0.University0.edu/AssociateProfessor0',x), takesCourse (y, x), type (x, 'Course'), type (y, 'UndergraduateStudent').
SELECT t.subject AS x, tc.subject AS y
FROM type t
JOIN takesCourse tc ON t.subject = tc.object
JOIN type t2 ON tc.subject = t2.subject
JOIN teacherOf tof ON tof.object = t.subject
WHERE t.object = 'Course'
AND t2.object = 'UndergraduateStudent'
AND tof.subject = 'http://www.Department0.University0.edu/AssociateProfessor0';