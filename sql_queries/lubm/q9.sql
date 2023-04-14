-- out (x,y,z) : - type (x, 'UndergraduateStudent'), type (y, 'Course'), type (z, 'AssistantProfessor'), advisor (x,z), teacherOf (z,y), takesCourse (x, y) .
SELECT t.subject AS x, tof.object AS y, a.object AS z
FROM type t
JOIN advisor a ON t.subject = a.subject
JOIN type t2 ON a.object = t2.subject
JOIN teacherOf tof ON a.object = tof.subject
JOIN type t3 ON tof.object = t3.subject
JOIN takesCourse tc ON tc.subject = t.subject AND tc.object = t3.subject
WHERE t.object = 'UndergraduateStudent'
AND t2.object = 'AssistantProfessor'
AND t3.object = 'Course';