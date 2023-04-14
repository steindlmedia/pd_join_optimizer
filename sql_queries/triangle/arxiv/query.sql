-- out(x, y, z) :- arxiv(x, y), arxiv(y, z), arxiv(z, x).
SELECT CAST(a.x AS BIGINT), CAST(a.y AS BIGINT), CAST(b.y AS BIGINT) AS z
FROM arxiv a
JOIN arxiv b ON a.y = b.x
JOIN arxiv c ON b.y = c.x
WHERE a.x = c.y;