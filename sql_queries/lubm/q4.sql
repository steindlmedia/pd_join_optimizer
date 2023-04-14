-- out (x, y, z, w) : - worksFor (x, 'http://www.Department0.University0.edu'), name (x, y), emailAddress (x, w), telephone (x, z), type (x, 'AssociateProfessor').
SELECT t.subject AS x, n.object AS y, te.object AS z, e.object AS w
FROM type t
JOIN telephone te ON t.subject = te.subject
JOIN emailAddress e ON t.subject = e.subject
JOIN name n ON t.subject = n.subject
JOIN worksFor wf ON t.subject = wf.subject
WHERE t.object = 'AssociateProfessor'
AND wf.object = 'http://www.Department0.University0.edu';