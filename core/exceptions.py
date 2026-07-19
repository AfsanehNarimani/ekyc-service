class EKYCError(Exception):
    pass


class NoFaceDetectedError(EKYCError):
    pass


class MultipleFacesDetectedError(EKYCError):
    pass
