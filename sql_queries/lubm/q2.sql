-- out (x,y,z) : - memberOf (x, y), subOrganizationOf (y, z), undergraduateDegreeFrom (x,z), type (x, 'GraduateStudent'), type (y, 'Department'), type (z, 'University').
SELECT t.subject AS x, m.object AS y, t3.subject AS z
FROM type t
JOIN memberOf m ON m.subject = t.subject
JOIN type t2 ON m.object = t2.subject
JOIN subOrganizationOf so ON m.object = so.subject
JOIN type t3 ON so.object = t3.subject
JOIN undergraduateDegreeFrom u ON u.subject = t.subject AND u.object = t3.subject
WHERE t.object = 'GraduateStudent'
AND t2.object = 'Department'
AND t3.object = 'University';