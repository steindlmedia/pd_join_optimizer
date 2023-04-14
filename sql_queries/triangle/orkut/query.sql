-- out(x, y, z) :- orkut(x, y), orkut(y, z), orkut(z, x).
SELECT CAST(a.x AS BIGINT), CAST(a.y AS BIGINT), CAST(b.y AS BIGINT) AS z
FROM orkut a
JOIN orkut b ON a.y = b.x
JOIN orkut c ON b.y = c.x
WHERE a.x = c.y;