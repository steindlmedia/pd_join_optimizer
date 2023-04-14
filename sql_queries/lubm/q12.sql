-- out (x,y) : - worksFor (y, x), type (y, 'FullProfessor'), subOrganizationOf (x, 'University0'), type (x, 'Department').
SELECT t.subject AS x, wf.subject AS y
FROM type t
JOIN subOrganizationOf so ON t.subject = so.subject
JOIN worksFor wf ON t.subject = wf.object
JOIN type t2 ON wf.subject = t2.subject
WHERE t.object = 'Department'
AND so.object = 'http://www.University0.edu'
AND t2.object = 'FullProfessor';