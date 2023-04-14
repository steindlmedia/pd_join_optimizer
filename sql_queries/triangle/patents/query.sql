-- out(x, y, z) :- patents(x, y), patents(y, z), patents(z, x).
SELECT CAST(a.x AS BIGINT), CAST(a.y AS BIGINT), CAST(b.y AS BIGINT) AS z
FROM patents a
JOIN patents b ON a.y = b.x
JOIN patents c ON b.y = c.x
WHERE a.x = c.y;