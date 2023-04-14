-- out (x) : - type (x, 'UndergraduateStudent'), memberOf (x, 'http://www.Department0.University0.edu').
SELECT t.subject AS x
FROM type t
JOIN memberOf m ON t.subject = m.subject
WHERE t.object = 'UndergraduateStudent'
AND m.object = 'http://www.Department0.University0.edu';