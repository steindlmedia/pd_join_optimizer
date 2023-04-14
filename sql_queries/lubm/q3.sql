-- out (x) : - type (x, 'Publication'), publicationAuthor (x, 'http://www.Department0.University0.edu/AssistantProfessor0').
SELECT t.subject AS x
FROM type t
JOIN publicationAuthor pa ON t.subject = pa.subject
WHERE t.object = 'Publication'
AND pa.object = 'http://www.Department0.University0.edu/AssistantProfessor0';