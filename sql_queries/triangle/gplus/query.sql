-- out(x, y, z) :- gplus(x, y), gplus(y, z), gplus(z, x).
SELECT CAST(a.x AS BIGINT), CAST(a.y AS BIGINT), CAST(b.y AS BIGINT) AS z
FROM gplus a
JOIN gplus b ON a.y = b.x
JOIN gplus c ON b.y = c.x
WHERE a.x = c.y;