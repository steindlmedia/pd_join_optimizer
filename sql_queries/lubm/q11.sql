-- out (x) : - type (x, 'ResearchGroup'), subOrganizationOf(x,'http://www.University0.edu').
SELECT t.subject AS x
FROM type t
JOIN subOrganizationOf so ON t.subject = so.subject
WHERE t.object = 'ResearchGroup'
AND so.object = 'http://www.University0.edu';