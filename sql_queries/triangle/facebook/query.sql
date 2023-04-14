-- out(x, y, z) :- facebook(x, y), facebook(y, z), facebook(z, x).
SELECT CAST(a.x AS BIGINT), CAST(a.y AS BIGINT), CAST(b.y AS BIGINT) AS z
FROM facebook a
JOIN facebook b ON a.y = b.x
JOIN facebook c ON b.y = c.x
WHERE a.x = c.y;