-- out (x,y,z) : - memberOf (x, y), emailAddress (x, z), type (x, 'UndergraduateStudent'), subOrganizationOf(y, 'http://www.University0.edu'), type (y, 'Department').
SELECT t.subject AS x, m.object AS y, e.object AS z
FROM type t
JOIN emailAddress e ON t.subject = e.subject
JOIN memberOf m ON t.subject = m.subject
JOIN type t2 ON m.object = t2.subject
JOIN subOrganizationOf so ON m.object = so.subject
WHERE t.object = 'UndergraduateStudent'
AND so.object = 'http://www.University0.edu'
AND t2.object = 'Department';