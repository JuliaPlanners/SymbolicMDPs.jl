(define (problem blocksworld-problem)
	(:domain blocksworld)
	(:objects
		a b - block
	)
	(:init
		(handempty)
		(ontable a)
		(ontable b)
		(clear a)
		(clear b)
	)
	(:goal (and
		;; ab
		(clear a) (ontable b) (on a b)
	))
)
