-- out (x) : - takesCourse (x, 'http://www.Department0.University0.edu/GraduateCourse0'), type (x, 'GraduateStudent').
SELECT t.subject AS x
FROM type t
JOIN takesCourse tc ON t.subject = tc.subject
WHERE t.object = 'GraduateStudent'
AND tc.object = 'http://www.Department0.University0.edu/GraduateCourse0';